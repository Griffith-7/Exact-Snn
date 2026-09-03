"""SP-03: exact multi-spike LIF with hard reset + saltation jump map.

    u' = (i - u)/tm,    i' = -i/ts,    hard reset u(t_f+) = u_reset at firing,
    inputs = delta-current impulses (t_k, w_k);  multiple spikes per neuron.

All propagation between events is closed-form, so spike times are exact (roots
of u(t) = theta found by bracketing + Newton; no ODE-solver error). Forward-mode
sensitivities of spike times w.r.t. input weights are propagated through the
reset using the saltation matrix.

Saltation across a hard reset (state x = (u, i), event g = u - theta,
reset map R(u, i) = (u_reset, i)):

    Xi = [[ u'_f+ / u'_f-,      0 ],
          [ 0,                  1 ]]

The i-component is reset-immune (the reset only touches u), so its row is the
identity and there is no coupling from s_u into s_i.  With u'_f+ = (i_f - u_reset)/tm
and u'_f- = (i_f - theta)/tm:

    Xi_uu = (i_f - u_reset)/(i_f - theta)   (= u'_f+/u'_f-, the scalar EventProp jump),

so the continuous part of the state derivative jumps as

    s_u^+ = Xi_uu * s_u^-,        s_i^+ = s_i^-,

and each spike time satisfies  dt_f/dw = -s_u(t_f^-) / u'(t_f^-).

Public API:
    ResetLIF.run()              - forward (all spike times)
    ResetLIF.run_with_state()   - forward + du/dt at each spike
    ResetLIF.sensitivity()      - dt_k/dw for one weight (scalar)
    ResetLIF.sensitivity_all()  - dt_k/dw for ALL weights (vectorized)
    ResetLIF.state_at()         - (u,i,s_u,s_i) at a fixed time

Optional companion module (lazy opt-in) -- a scalar-level, dependency-free
reference solver for the multi-spike saltation jump map. Import it only if
needed:
    from exact_snn.reset import ResetLIF
"""
from __future__ import annotations

import math


try:
    import numpy as _np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


class ResetLIF:
    def __init__(self, tm: float = 15.0, ts: float = 4.0, theta: float = 1.0,
                 u_rest: float = 0.0, u_reset: float = 0.0,
                 dt_scan: float = 0.05) -> None:
        """Initialise LIF neuron parameters.

        Args:
            tm: Membrane time constant.
            ts: Synaptic time constant.
            theta: Spike threshold.
            u_rest: Resting membrane potential.
            u_reset: Reset membrane potential after a spike.
            dt_scan: Coarse scan step for bracketing threshold crossings.
        """
        self.tm = float(tm)
        self.ts = float(ts)
        self.theta = float(theta)
        self.u_rest = float(u_rest)
        self.u_reset = float(u_reset)
        self.dt_scan = float(dt_scan)

    # ---- closed-form propagation between events -------------------------
    def _propagate(self, u0, i0, dt):
        """Exact (u, i) after time dt with no events."""
        if dt <= 0.0:
            return u0, i0
        a, b = 1.0 / self.tm, 1.0 / self.ts
        i = i0 * math.exp(-b * dt)
        u = (u0 * math.exp(-a * dt)
             + i0 * (math.exp(-b * dt) - math.exp(-a * dt)) / (1.0 - self.tm / self.ts))
        return u, i

    def _up(self, u, i):
        """u'(t) at state (u, i)."""
        return (i - u) / self.tm

    def _find_first_crossing(self, u0, i0, t0, t1):
        """First root of u(t) - theta in (t0, t1), returned as (t_f, i_f), or
        None. Bracketing scan + bisection + Newton polish (exact)."""
        span = t1 - t0
        n = max(2, int(round(span / self.dt_scan)))
        dt = span / n
        prev_u = u0
        for k in range(1, n + 1):
            tg = t0 + k * dt
            ug, ig = self._propagate(u0, i0, tg - t0)
            if (prev_u < self.theta <= ug) or (prev_u > self.theta >= ug):
                a, b = tg - dt, tg
                ua = self._propagate(u0, i0, a - t0)[0] - self.theta
                for _ in range(80):
                    m = 0.5 * (a + b)
                    um = self._propagate(u0, i0, m - t0)[0] - self.theta
                    if ua * um <= 0.0:
                        b, um_b = m, um
                    else:
                        a, ua = m, um
                m = 0.5 * (a + b)
                for _ in range(12):
                    um, im = self._propagate(u0, i0, m - t0)
                    dm = self._up(um, im)
                    if abs(dm) < 1e-12:
                        break
                    nm = m - (um - self.theta) / dm
                    nm = max(a, min(b, nm))
                    if abs(nm - m) < 1e-14:
                        m = nm
                        break
                    m = nm
                tf, if_ = self._propagate(u0, i0, m - t0)
                return m, if_
            prev_u = ug
        return None

    # ---- forward --------------------------------------------------------
    def run(self, inputs: list[tuple[float, float]], t_end: float = 200.0) -> list[float]:
        """inputs: iterable of (t, w). Returns the list of fire times."""
        evs = sorted((float(t), float(w)) for (t, w) in inputs)
        fires = []
        u, i = self.u_rest, 0.0
        t = 0.0
        for tk, wk in evs + [(t_end, 0.0)]:
            if tk <= t:
                i += wk
                continue
            while t < tk - 1e-12:
                res = self._find_first_crossing(u, i, t, tk)
                if res is None:
                    break
                tf, i_f = res
                fires.append(tf)
                u, i = self.u_reset, i_f
                t = tf
            u, i = self._propagate(u, i, tk - t)
            i += wk
            t = tk
        return fires

    # ---- forward-mode sensitivity (with saltation at each reset) --------
    def sensitivity(self, inputs: list[tuple[float, float]], w_idx: int, t_end: float = 200.0) -> tuple[list[float], list[float]]:
        """d(spike_time)/d(w_{w_idx}) for every spike, via the saltation.

        Returns (fires, dtdw). The variational (s_u, s_i) is propagated by the
        same closed form as (u, i); at each input jump s_i += 1 (dw/dw); at each
        reset the saltation Xi is applied. Grazing spikes (|u'| ~ 0) return
        +/-inf and are flagged, never NaN.
        """
        evs = sorted((float(t), float(w), m) for m, (t, w) in enumerate(inputs))
        fires, dtdw = [], []
        u, i = self.u_rest, 0.0
        s_u, s_i = 0.0, 0.0
        t = 0.0
        for tk, wk, m in evs + [(t_end, 0.0, -1)]:
            if tk <= t:
                i += wk
                if m == w_idx:
                    s_i += 1.0
                continue
            while t < tk - 1e-12:
                res = self._find_first_crossing(u, i, t, tk)
                if res is None:
                    break
                tf, i_f = res
                s_uf = self._propagate(s_u, s_i, tf - t)[0]
                s_if = self._propagate(s_u, s_i, tf - t)[1]
                up_f = (i_f - self.theta) / self.tm
                if abs(up_f) > 1e-10:
                    dtf = -s_uf / up_f
                else:
                    dtf = math.copysign(math.inf, -s_uf)
                fires.append(tf)
                dtdw.append(dtf)
                den = i_f - self.theta
                if abs(den) > 1e-12:
                    s_u_new = (i_f - self.u_reset) / den * s_uf
                    s_i_new = s_if
                else:
                    s_u_new = math.copysign(math.inf, s_uf)
                    s_i_new = math.inf
                u, i = self.u_reset, i_f
                s_u, s_i = s_u_new, s_i_new
                t = tf
            u, i = self._propagate(u, i, tk - t)
            s_u, s_i = self._propagate(s_u, s_i, tk - t)
            i += wk
            if m == w_idx:
                s_i += 1.0
            t = tk
        return fires, dtdw

    def run_with_state(self, inputs: list[tuple[float, float]], t_end: float = 200.0) -> tuple[list[float], list[float]]:
        """Forward that also returns du/dt at each spike.

        Returns (fires, up_at_fires).  Each entry in up_at_fires is the
        membrane derivative u'(t_f^-) just BEFORE the reset -- i.e. the
        IFT denominator u'_f = (i_f - theta)/tm.
        """
        evs = sorted((float(t), float(w)) for (t, w) in inputs)
        fires, ups = [], []
        u, i = self.u_rest, 0.0
        t = 0.0
        for tk, wk in evs + [(t_end, 0.0)]:
            if tk <= t:
                i += wk
                continue
            while t < tk - 1e-12:
                res = self._find_first_crossing(u, i, t, tk)
                if res is None:
                    break
                tf, i_f = res
                fires.append(tf)
                ups.append((i_f - self.theta) / self.tm)
                u, i = self.u_reset, i_f
                t = tf
            u, i = self._propagate(u, i, tk - t)
            i += wk
            t = tk
        return fires, ups

    # ---- ALL-weight sensitivity (vectorised over weights) ----------------
    def sensitivity_all(self, inputs: list[tuple[float, float]], t_end: float = 200.0) -> tuple[list[float], list[list[float]]]:
        """d(spike_k)/d(w_m) for ALL weights m in a single pass.

        Returns (fires, dtdw_matrix) where dtdw_matrix[k][m] is the
        sensitivity of the k-th spike time to the m-th weight.  Uses
        vectorised variational states: one (s_u, s_i) pair per weight,
        all sharing the same forward trajectory through events + resets.

        Cost: O(n_weights * n_events) per call -- one forward pass, not
        n_weights separate passes as sensitivity().
        """
        n_w = len(inputs)
        evs = sorted((float(t), float(w), m) for m, (t, w) in enumerate(inputs))
        fires = []
        dtdw_matrix = []

        u, i = self.u_rest, 0.0
        s_u = [0.0] * n_w
        s_i = [0.0] * n_w
        t = 0.0

        for tk, wk, m in evs + [(t_end, 0.0, -1)]:
            if tk <= t:
                i += wk
                s_i[m] += 1.0
                continue
            while t < tk - 1e-12:
                res = self._find_first_crossing(u, i, t, tk)
                if res is None:
                    break
                tf, i_f = res
                dt_fire = tf - t
                s_uf = [0.0] * n_w
                s_if = [0.0] * n_w
                for k in range(n_w):
                    s_uf[k] = self._propagate(s_u[k], s_i[k], dt_fire)[0]
                    s_if[k] = self._propagate(s_u[k], s_i[k], dt_fire)[1]
                up_f = (i_f - self.theta) / self.tm
                dtdw = [0.0] * n_w
                for k in range(n_w):
                    if abs(up_f) > 1e-10:
                        dtdw[k] = -s_uf[k] / up_f
                    else:
                        dtdw[k] = math.copysign(math.inf, -s_uf[k])
                fires.append(tf)
                dtdw_matrix.append(dtdw)
                den = i_f - self.theta
                if abs(den) > 1e-12:
                    Xi_uu = (i_f - self.u_reset) / den
                    for k in range(n_w):
                        s_u[k] = Xi_uu * s_uf[k]
                        s_i[k] = s_if[k]
                else:
                    for k in range(n_w):
                        s_u[k] = math.copysign(math.inf, s_uf[k])
                        s_i[k] = math.inf
                u, i = self.u_reset, i_f
                t = tf
            u, i = self._propagate(u, i, tk - t)
            for k in range(n_w):
                s_u[k], s_i[k] = self._propagate(s_u[k], s_i[k], tk - t)
            i += wk
            s_i[m] += 1.0
            t = tk
        return fires, dtdw_matrix

    # ---- first-spike only (vectorised, early exit) ----------------------
    def sensitivity_first_spike(self, inputs: list[tuple[float, float]], t_end: float = 200.0) -> tuple[float | None, _np.ndarray]:
        """d(first_spike)/d(w_m) for ALL weights — numpy-vectorized, early exit.

        Returns (fire_time, dtdw_array) or (None, zeros) if no spike.
        ~n_w x faster than sensitivity_all because the inner weight loop
        uses numpy array ops instead of Python list loops, and processing
        stops at the first spike (no unnecessary event traversal).
        """
        n_w = len(inputs)
        evs = sorted((float(t), float(w), m) for m, (t, w) in enumerate(inputs))

        u, ci = self.u_rest, 0.0
        s_u = _np.zeros(n_w, dtype=_np.float64)
        s_i = _np.zeros(n_w, dtype=_np.float64)
        t = 0.0
        a, br = 1.0 / self.tm, 1.0 / self.ts
        fac = 1.0 / (1.0 - self.tm / self.ts)

        for tk, wk, m in evs + [(t_end, 0.0, -1)]:
            if tk <= t:
                ci += wk
                s_i[m] += 1.0
                continue
            while t < tk - 1e-12:
                res = self._find_first_crossing(u, ci, t, tk)
                if res is None:
                    break
                tf, i_f = res
                dt_f = tf - t
                ea = math.exp(-a * dt_f)
                eb = math.exp(-br * dt_f)
                cp = (eb - ea) * fac
                s_uf = s_u * ea + s_i * cp
                up_f = (i_f - self.theta) / self.tm
                if abs(up_f) > 1e-10:
                    return tf, -s_uf / up_f
                return tf, _np.full(n_w, math.copysign(math.inf, -float(s_uf[0])))
            dt_e = tk - t
            ea = math.exp(-a * dt_e)
            eb = math.exp(-br * dt_e)
            cp = (eb - ea) * fac
            u, s_u = u * ea + ci * cp, s_u * ea + s_i * cp
            ci, s_i = ci * eb, s_i * eb
            ci += wk
            s_i[m] += 1.0
            t = tk
        return None, _np.zeros(n_w, dtype=_np.float64)

    # ---- fixed-time state + sensitivity ---------------------------------
    def state_at(self, inputs: list[tuple[float, float]], t_eval: float, w_idx: int = 0, use_saltation: bool = True, t_end: float = 200.0) -> tuple[float, float, float, float]:
        """(u, i, s_u, s_i) at a fixed time t_eval (no crossing at t_eval),
        with the saltation applied at resets -- used for the fixed-time FD
        check of the jump map. With use_saltation=False the variationals are
        propagated through the reset as the identity (the no-jump control)."""
        evs = sorted((float(t), float(w), m) for m, (t, w) in enumerate(inputs))
        u, i = self.u_rest, 0.0
        s_u, s_i = 0.0, 0.0
        t = 0.0
        for tk, wk, m in evs + [(t_end, 0.0, -1)]:
            if tk <= t:
                i += wk
                if m == w_idx:
                    s_i += 1.0
                continue
            if t >= t_eval:
                break
            stop = min(tk, t_eval)
            while t < stop - 1e-12:
                res = self._find_first_crossing(u, i, t, stop)
                if res is None:
                    break
                tf, i_f = res
                s_uf = self._propagate(s_u, s_i, tf - t)[0]
                s_if = self._propagate(s_u, s_i, tf - t)[1]
                den = i_f - self.theta
                if use_saltation:
                    if abs(den) > 1e-12:
                        s_u = (i_f - self.u_reset) / den * s_uf
                        s_i = s_if
                    else:
                        s_u = math.copysign(math.inf, s_uf)
                        s_i = math.inf
                else:
                    s_u, s_i = s_uf, s_if
                u, i = self.u_reset, i_f
                t = tf
            u, i = self._propagate(u, i, stop - t)
            s_u, s_i = self._propagate(s_u, s_i, stop - t)
            t = stop
            if t >= t_eval:
                break
            i += wk
            if m == w_idx:
                s_i += 1.0
        return u, i, s_u, s_i
