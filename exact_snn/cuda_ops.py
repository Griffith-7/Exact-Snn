"""SP-05: optional CUDA extension for the exact TTFS first-spike root-solve.

The torch forward (`_forward_layer_torch`) computes the grid membrane `U`, then
runs `n_bisect` bisection steps + `n_newton` Newton steps in *Python tensor
loops* (many small kernel launches), plus a golden-section peak search for
near-threshold silent neurons. The CUDA extension moves those cheap-but-looping
scalar root-solves into a single fused kernel: one thread per (neuron, sample)
does the grid scan, bisection, Newton refinement and (if needed) the peak
search. `U` is built in a transposed (n, G, B) layout for coalesced scans, and
each block tiles one weight row + a batch slice of `t_prev` into shared memory
so the exact `u_at`/`du_at` recomputations never touch global memory.

The extension is loaded lazily via `torch.utils.cpp_extension.load_inline` and
is *optional*: if it cannot be built (no nvcc, no MSVC on Windows) the package
falls back silently to the torch path. `keep_installed` leaves the compiled
`.pyd` on disk in the torch extensions cache, so subsequent imports are cheap.

Usage
-----
    from exact_snn import cuda_ops
    cuda_ops.available()      # True once the extension built successfully
    cuda_ops.status()         # human-readable backend + any build error
    cuda_ops.set_enabled(False)  # force the torch path even if built

The dispatch hooks live in `_ExactTTFSLayerFn.forward` in `exact_snn/__init__.py`,
so `ExactTTFSLinear`, `ExactTTFSConv2d`, and `ExactRecurrent` (single-spike TTFS)
automatically use the CUDA root-solve when the input is on CUDA.
"""
from __future__ import annotations

import os
import platform

import torch

_EXT = None
_ATTEMPTED = False
_ENABLED = True
_ERR: str | None = None
_K_FN = None

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

template<typename T>
__device__ __forceinline__ T K_dev(T d, T tm, T ts, bool alpha, T k_peak){
    if (d != d) d = (T)0;       // NaN -> 0 (matches torch _K)
    if (d < (T)0) d = (T)0;
    if (alpha){
        return (d / tm) * exp((T)1 - d / tm) / k_peak;
    } else {
        return (exp(-d / tm) - exp(-d / ts)) / (tm - ts) / k_peak;
    }
}

template<typename T>
__device__ __forceinline__ T Kd_dev(T d0, T tm, T ts, bool alpha, T k_peak){
    bool positive = (d0 > (T)0);
    T d = d0 < (T)0 ? (T)0 : d0;
    T val;
    if (alpha){
        val = ((T)1 - d / tm) * exp((T)1 - d / tm) / (tm * k_peak);
    } else {
        val = (-exp(-d / tm) / tm + exp(-d / ts) / ts) / (tm - ts) / k_peak;
    }
    return positive ? val : (T)0;
}

#define TTFS_TILE 32

// exact membrane and its time-derivative from shared-memory tiles: the block
// holds one weight row (s_W) and a (n_in+1) x TTFS_TILE slice of t_prev, so the
// Newton/peak-search u_at evaluations cost shared-memory traffic only instead
// of a fresh O(n_in) global read per thread per step.
template<typename T>
__device__ __forceinline__ T u_at_sm(const T* s_W, const T* s_tp, int n_in,
                                     int loc, T t_bias, T tm, T ts,
                                     bool alpha, T k_peak, T t){
    T u = s_W[n_in] * K_dev(t - t_bias, tm, ts, alpha, k_peak);
    for (int i = 0; i < n_in; i++){
        u += s_W[i] * K_dev(t - s_tp[i * TTFS_TILE + loc], tm, ts, alpha, k_peak);
    }
    return u;
}

template<typename T>
__device__ __forceinline__ T du_at_sm(const T* s_W, const T* s_tp, int n_in,
                                      int loc, T t_bias, T tm, T ts,
                                      bool alpha, T k_peak, T t){
    T du = s_W[n_in] * Kd_dev(t - t_bias, tm, ts, alpha, k_peak);
    for (int i = 0; i < n_in; i++){
        du += s_W[i] * Kd_dev(t - s_tp[i * TTFS_TILE + loc], tm, ts, alpha, k_peak);
    }
    return du;
}

// Linear interpolation of the grid membrane U (n_cur, G, B): `Ubase` points at
// the (n, g, b) block whose `grid` is read with the batch index on the last
// axis so warp co-loads of U stay coalesced.
template<typename T>
__device__ __forceinline__ T interp_grid(const T* Ubase, const T* grid, int b,
                                         int B, int G, T m){
    T t0 = grid[0], t1 = grid[G - 1];
    if (m < t0) m = t0; else if (m > t1) m = t1;
    T pos = (m - t0) / (t1 - t0) * (T)(G - 1);
    long lo = (long)pos;
    if (lo < 0) lo = 0; if (lo > G - 1) lo = G - 1;
    long hi = lo + 1; if (hi > G - 1) hi = G - 1;
    T frac = pos - (T)lo;
    if (frac < (T)0) frac = (T)0; if (frac > (T)1) frac = (T)1;
    const T vlo = Ubase[(size_t)lo * B + b];
    const T vhi = Ubase[(size_t)hi * B + b];
    return vlo + frac * (vhi - vlo);
}

template<typename T>
__global__ void exact_forward_kernel(
    const T* U, const T* W, const T* t_prev, const T* grid,
    const int n_cur, const int B, const int n_in, const int G,
    T t_bias, T theta, T tm, T ts, bool alpha, T k_peak,
    int n_bisect, int n_newton, T peak_tol,
    T* t_post, T* up){
    extern __shared__ char _smem[];
    T* s_W  = (T*)(void*)_smem;
    T* s_tp = (T*)(void*)(_smem + (n_in + 1) * sizeof(T));

    const int tid = threadIdx.x;
    const int n   = blockIdx.y;
    const int b0  = blockIdx.x * TTFS_TILE;
    const T* Ubase = U + (size_t)n * G * B;

    for (int i = tid; i <= n_in; i += TTFS_TILE)
        s_W[i] = W[n * (n_in + 1) + i];
    for (int i = tid; i <= n_in; i += TTFS_TILE)
        for (int c = 0; c < TTFS_TILE; c++){
            const int bb = b0 + c;
            s_tp[i * TTFS_TILE + c] = (bb < B) ? t_prev[i * B + bb] : (T)0;
        }
    __syncthreads();

    const int b = b0 + tid;
    if (b >= B) return;

    int first = -1;
    for (int g = 0; g < G; g++){
        if (Ubase[(size_t)g * B + b] >= theta){ first = g; break; }
    }

    if (first >= 0){
        T tf;
        if (first == 0){
            tf = grid[0];
        } else {
            T a = grid[first - 1];
            T b_ = grid[first];
            T fa = Ubase[(size_t)(first - 1) * B + b] - theta;
            for (int it = 0; it < n_bisect; it++){
                T m = (T)0.5 * (a + b_);
                T fm = interp_grid(Ubase, grid, b, B, G, m) - theta;
                if (fa * fm <= (T)0){ b_ = m; } else { a = m; fa = fm; }
            }
            T m = (T)0.5 * (a + b_);
            for (int it = 0; it < n_newton; it++){
                T um = u_at_sm(s_W, s_tp, n_in, tid, t_bias, tm, ts, alpha, k_peak, m) - theta;
                T dum = du_at_sm(s_W, s_tp, n_in, tid, t_bias, tm, ts, alpha, k_peak, m);
                bool safe = (dum > (T)1e-10);
                T dn = safe ? dum : (T)1;
                T nm = m - um / dn;
                if (nm < a) nm = a; else if (nm > b_) nm = b_;
                if (safe) m = nm;
            }
            tf = m;
        }
        t_post[(size_t)n * B + b] = tf;
        up[(size_t)n * B + b] = du_at_sm(s_W, s_tp, n_in, tid, t_bias, tm, ts, alpha, k_peak, tf);
    } else {
        T t_post_val = INFINITY;
        T up_val = (T)0;
        T u_max = -INFINITY;
        int imax = 0;
        for (int g = 0; g < G; g++){
            const T ug = Ubase[(size_t)g * B + b];
            if (ug > u_max){ u_max = ug; imax = g; }
        }
        if (u_max >= theta - peak_tol){
            int lo_i = imax - 1 < 0 ? 0 : imax - 1;
            int hi_i = imax + 1 > G - 1 ? G - 1 : imax + 1;
            T lo = grid[lo_i], hi = grid[hi_i];
            const T gr = (sqrt((T)5) - (T)1) / (T)2;
            T c = hi - gr * (hi - lo);
            T d = lo + gr * (hi - lo);
            for (int it = 0; it < 12; it++){
                T uc = interp_grid(Ubase, grid, b, B, G, c);
                T ud = interp_grid(Ubase, grid, b, B, G, d);
                if (uc > ud){ hi = d; } else { lo = c; }
                c = hi - gr * (hi - lo);
                d = lo + gr * (hi - lo);
            }
            T t_peak = (T)0.5 * (lo + hi);
            T u_peak = interp_grid(Ubase, grid, b, B, G, t_peak);
            if (u_peak >= theta){
                T a2 = (T)0;
                T b2 = t_peak;
                T fa2 = interp_grid(Ubase, grid, b, B, G, a2) - theta;
                for (int it = 0; it < n_bisect; it++){
                    T m2 = (T)0.5 * (a2 + b2);
                    T fm2 = interp_grid(Ubase, grid, b, B, G, m2) - theta;
                    if (fa2 * fm2 <= (T)0){ b2 = m2; } else { a2 = m2; fa2 = fm2; }
                }
                T m2 = (T)0.5 * (a2 + b2);
                for (int it = 0; it < n_newton; it++){
                    T um = u_at_sm(s_W, s_tp, n_in, tid, t_bias, tm, ts, alpha, k_peak, m2) - theta;
                    T dum = du_at_sm(s_W, s_tp, n_in, tid, t_bias, tm, ts, alpha, k_peak, m2);
                    bool safe = (dum > (T)1e-10);
                    T dn = safe ? dum : (T)1;
                    T nm = m2 - um / dn;
                    if (nm < a2) nm = a2; else if (nm > b2) nm = b2;
                    if (safe) m2 = nm;
                }
                t_post_val = m2;
                up_val = du_at_sm(s_W, s_tp, n_in, tid, t_bias, tm, ts, alpha, k_peak, m2);
            }
        }
        t_post[(size_t)n * B + b] = t_post_val;
        up[(size_t)n * B + b] = up_val;
    }
}

template<typename T>
void launch(const T* U, const T* W, const T* t_prev, const T* grid,
            int n_cur, int B, int n_in, int G,
            T t_bias, T theta, T tm, T ts, bool alpha, T k_peak,
            int n_bisect, int n_newton, T peak_tol,
            T* t_post, T* up){
    dim3 block(TTFS_TILE);
    dim3 grid_dim((B + TTFS_TILE - 1) / TTFS_TILE, n_cur);
    size_t sbytes = (size_t)(n_in + 1) * (TTFS_TILE + 1) * sizeof(T);
    auto kern = exact_forward_kernel<T>;
    cudaError_t att = cudaFuncSetAttribute(
        kern, cudaFuncAttributeMaxDynamicSharedMemorySize, sbytes);
    TORCH_CHECK(att == cudaSuccess,
                "cudaFuncSetAttribute failed (shared memory too large?): ",
                cudaGetErrorString(att));
    kern<<<grid_dim, block, sbytes>>>(
        U, W, t_prev, grid, n_cur, B, n_in, G,
        t_bias, theta, tm, ts, alpha, k_peak,
        n_bisect, n_newton, peak_tol, t_post, up);
}

std::vector<torch::Tensor> exact_forward(
    torch::Tensor W, torch::Tensor t_prev,
    double t_bias, double theta, double tm, double ts,
    bool alpha, double k_peak, torch::Tensor grid,
    int64_t n_bisect, int64_t n_newton, double peak_tol,
    torch::Tensor U){
    TORCH_CHECK(W.is_cuda(), "W must be CUDA");
    TORCH_CHECK(U.is_cuda(), "U must be CUDA");
    TORCH_CHECK(U.dim() == 3, "U must be (n_cur, G, B)");
    int n_cur = (int)U.size(0);
    int G     = (int)U.size(1);
    int B     = (int)U.size(2);
    int n_in  = (int)(W.size(1) - 1);
    auto opts = W.options();
    auto t_post = torch::full({n_cur, B}, double(INFINITY), opts);
    auto up     = torch::zeros({n_cur, B}, opts);

    auto Wc = W.contiguous();
    auto tp = t_prev.contiguous();
    auto gc = grid.contiguous();
    auto Uc = U.contiguous();

    if (W.scalar_type() == torch::kDouble){
        launch<double>(Uc.data_ptr<double>(), Wc.data_ptr<double>(),
                       tp.data_ptr<double>(), gc.data_ptr<double>(),
                       n_cur, B, n_in, G,
                       (double)t_bias, (double)theta, (double)tm, (double)ts,
                       alpha, (double)k_peak, (int)n_bisect, (int)n_newton,
                       (double)peak_tol,
                       t_post.data_ptr<double>(), up.data_ptr<double>());
    } else if (W.scalar_type() == torch::kFloat){
        launch<float>(Uc.data_ptr<float>(), Wc.data_ptr<float>(),
                      tp.data_ptr<float>(), gc.data_ptr<float>(),
                      n_cur, B, n_in, G,
                      (float)t_bias, (float)theta, (float)tm, (float)ts,
                      alpha, (float)k_peak, (int)n_bisect, (int)n_newton,
                      (float)peak_tol,
                      t_post.data_ptr<float>(), up.data_ptr<float>());
    } else {
        TORCH_CHECK(false, "exact_forward supports float32/float64 only");
    }
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    return {t_post, up};
}
"""

_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> exact_forward(
    torch::Tensor W, torch::Tensor t_prev,
    double t_bias, double theta, double tm, double ts,
    bool alpha, double k_peak, torch::Tensor grid,
    int64_t n_bisect, int64_t n_newton, double peak_tol,
    torch::Tensor U);
"""


def _build_flags():
    """extra_cuda_cflags for the MSVC/traditional-preprocessor CCCL issue.

    CUDA >=12's bundled CCCL (libcu++) hard-errors under cl's traditional
    preprocessor (common after nvcc 12.8 / VS 17.8+). The fix is cl's standard
    conforming preprocessor via -Xcompiler /Zc:preprocessor.
    """
    flags = []
    if platform.system() == "Windows":
        flags += ["-Xcompiler", "/Zc:preprocessor",
                  "-DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING"]
    return flags


def _configure_toolchain():
    """In-process MSVC + Windows-SDK environment for torch cpp_extension.

    On a bare shell / IDE launch, INCLUDE, LIB and Path do not contain the
    MSVC toolchain entries torch's JIT needs to compile CUDA sources. This
    prepends them (from vswhere + the Windows Kits layout) so `load_inline`
    works without a separately opened "x64 Native Tools" prompt. No-op on
    non-Windows hosts.
    """
    if platform.system() != "Windows":
        return
    import glob
    import subprocess

    def _find_vs() -> str:
        for cand in (os.environ.get("VSINSTALLDIR"), ""):
            if cand and os.path.isdir(cand):
                return cand
        vswhere = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
        if os.path.exists(vswhere):
            try:
                out = subprocess.run(
                    [vswhere, "-latest", "-products", "*",
                     "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                     "-property", "installationPath"],
                    capture_output=True, text=True, timeout=30).stdout.strip()
                if out and os.path.isdir(out):
                    return out
            except Exception:  # noqa: BLE001 - fall through to defaults
                pass
        for base in (r"C:\Program Files\Microsoft Visual Studio",
                     r"C:\Program Files (x86)\Microsoft Visual Studio"):
            if os.path.isdir(base):
                subs = sorted(os.listdir(base))
                for s in reversed(subs):
                    p = os.path.join(base, s, "Community")
                    if os.path.isdir(p):
                        return p
                    p = os.path.join(base, s, "BuildTools")
                    if os.path.isdir(p):
                        return p
        return ""

    vs = _find_vs()
    if not vs or not os.path.isdir(vs):
        return

    msdev = sorted(glob.glob(os.path.join(vs, "VC", "Tools", "MSVC", "*")))
    msvc = os.path.basename(msdev[-1]) if msdev else ""
    ks_root = r"C:\Program Files (x86)\Windows Kits\10"
    ks = sorted(glob.glob(os.path.join(ks_root, "Include", "10.*")))
    sdk_ver = os.path.basename(ks[-1]) if ks else ""
    if not msvc and not sdk_ver:
        return

    tk = os.path.join(vs, "VC", "Tools", "MSVC", msvc)
    need_inc = []
    need_lib = []
    need_path = []
    if msvc:
        need_inc += [os.path.join(tk, "include"),
                     os.path.join(tk, "atlmfc", "include")]
        need_lib += [os.path.join(tk, "lib", "x64"),
                     os.path.join(tk, "atlmfc", "lib", "x64")]
        need_path += [os.path.join(tk, "bin", "Hostx64", "x64")]
    if sdk_ver:
        for sub in ("um", "ucrt", "shared"):
            need_inc.append(os.path.join(ks_root, "Include", sdk_ver, sub))
        need_lib += [os.path.join(ks_root, "Lib", sdk_ver, "um", "x64"),
                     os.path.join(ks_root, "Lib", sdk_ver, "ucrt", "x64")]
        need_path += [os.path.join(ks_root, "bin", sdk_ver, "x64")]

    def _prepend(name, entries):
        cur = os.environ.get(name)
        parts = [e for e in entries
                 if e and os.path.isdir(e) and cur is not None and e not in cur.split(os.pathsep)] \
            if cur is not None else [e for e in entries if e and os.path.isdir(e)]
        if parts:
            os.environ[name] = os.pathsep.join(parts) + (os.pathsep + cur if cur else "")

    _prepend("INCLUDE", need_inc)
    _prepend("LIB", need_lib)
    _prepend("Path", need_path)


def _load():
    global _EXT, _ATTEMPTED, _ERR
    if _EXT is not None:
        return True
    if _ATTEMPTED:
        return False
    _ATTEMPTED = True
    if not torch.cuda.is_available():
        _ERR = "CUDA is not available on this machine"
        return False
    try:
        from torch.utils.cpp_extension import load_inline
        _configure_toolchain()
        _EXT = load_inline(
            name="exact_snn_cuda",
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=["exact_forward"],
            extra_cuda_cflags=_build_flags(),
            verbose=(os.environ.get("EXACT_SNN_CUDA_VERBOSE") == "1"),
        )
        return True
    except Exception as e:  # noqa: BLE001 - any build/runtime error -> fallback
        _ERR = repr(e)
        _EXT = None
        return False


def available() -> bool:
    """True if the CUDA extension is built and loadable."""
    return _load()


def backend_error():
    """Human-readable reason why the CUDA backend is unavailable (or None)."""
    _load()
    return _ERR


def set_enabled(value: bool) -> None:
    """Force the torch path globally even if the CUDA extension is available."""
    global _ENABLED
    _ENABLED = bool(value)


def is_enabled() -> bool:
    _load()
    return _ENABLED and _EXT is not None


def status() -> str:
    _load()
    if not torch.cuda.is_available():
        return "cuda: no CUDA device (using torch path)"
    if _EXT is None:
        return f"cuda: extension not built (using torch path). {_ERR}"
    return "cuda: native root-solve enabled" if _ENABLED else "cuda: native built but disabled (torch path)"


def cuda_forward(W, t_prev, t_bias, theta, grid, tm, ts, alpha, k_peak,
                 n_bisect=15, n_newton=8, peak_tol=1e-2):
    """CUDA first-spike forward, shape/semantics identical to
    `_forward_layer_torch`. Returns (t_post, up).

    U is built with the same single-matmul grid construction as the torch path
    so the kernel operates on an identical membrane grid.
    """
    if not _load():
        raise RuntimeError(
            f"CUDA extension unavailable: {_ERR} (use the torch path instead)")
    from exact_snn import _K
    global _K_FN
    if _K_FN is None:
        _K_FN = _K
    n_cur, n_inp = W.shape
    n_in = n_inp - 1
    B = t_prev.shape[1]
    G = grid.numel()
    # U is built in (n_cur, G, B) (batch on the last axis) so the kernel's grid
    # scan is coalesced across a warp.
    g = grid.view(1, -1, 1)
    t_data = t_prev[:n_in]
    D = g - t_data.unsqueeze(1)
    K_vals = _K_FN(D, tm, ts, alpha, k_peak)
    U = (W[:, :n_in] @ K_vals.reshape(n_in, -1)).reshape(n_cur, G, B)
    U = U + W[:, n_in].view(n_cur, 1, 1) * _K_FN(g - t_bias, tm, ts, alpha, k_peak)
    return _EXT.exact_forward(
        W, t_prev, float(t_bias), float(theta), float(tm), float(ts),
        bool(alpha), float(k_peak), grid, int(n_bisect), int(n_newton),
        float(peak_tol), U)