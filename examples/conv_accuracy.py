import os, sys, time
import numpy as np
import torch
from torchvision import datasets, transforms

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from exact_snn import latency_encode, latency_cross_entropy
from exact_snn.extended import ExactTTFSConv2d, ExactTTFSLinear

DEVICE = torch.device("cpu")
DTYPE = torch.float32
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)


def load_mnist(n):
    ds = datasets.MNIST(root=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
                        train=True, download=True, transform=transforms.ToTensor())
    X = torch.stack([ds[i][0] for i in range(min(n, len(ds)))]).view(min(n, len(ds)), 784)
    y = torch.tensor([ds[i][1] for i in range(min(n, len(ds)))])
    return X.to(DEVICE, DTYPE), y.to(DEVICE)


def downsample(X, f=2):
    B = X.shape[0]
    Xg = X.view(B, 1, 28, 28)
    return torch.nn.functional.avg_pool2d(Xg, f, f)


class ConvSNN(torch.nn.Module):
    def __init__(self, conv, fc):
        super().__init__()
        self.conv, self.fc = conv, fc

    def forward(self, x):
        tc = self.conv(x)
        tcf = torch.where(torch.isfinite(tc), tc, torch.full_like(tc, 39.0))
        return self.fc(tcf.reshape(tcf.shape[0], -1).t())


class ConvStackSNN(torch.nn.Module):
    """Two stacked TTFS conv layers (stride-2 on the 2nd) -> FC."""

    def __init__(self, conv1, conv2, fc, H, W):
        super().__init__()
        self.conv1, self.conv2, self.fc = conv1, conv2, fc
        self.H, self.W = H, W

    def forward(self, x):
        t1 = self.conv1(x)
        t1s = torch.where(torch.isfinite(t1), t1, torch.full_like(t1, 39.0))
        t2 = self.conv2(t1s)
        t2s = torch.where(torch.isfinite(t2), t2, torch.full_like(t2, 39.0))
        return self.fc(t2s.reshape(t2s.shape[0], -1).t())


def build(depth, N, CH):
    X, y = load_mnist(N)
    X14 = torch.nn.functional.avg_pool2d(X.view(N, 1, 28, 28), 2, 2)  # (N,1,14,14)
    if depth == 1:
        conv = ExactTTFSConv2d(1, CH, kernel_size=3, stride=1, padding=1,
                               t_max=40.0, dtype=DTYPE, device=DEVICE,
                               w_scale=0.35, bias_val=0.5, grid_pts=301)
        fc = ExactTTFSLinear(CH * 14 * 14, 10, t_max=40.0, dtype=DTYPE,
                             device=DEVICE, bias_val=1.5, grid_pts=301)
        return ConvSNN(conv, fc).to(DEVICE), latency_encode(X14, t_max=40.0), y
    if depth == "28":  # full 28x28 input, strided conv -> keeps 1568 FC features
        X28 = X.view(N, 1, 28, 28)  # (N,1,28,28) full resolution
        c1 = ExactTTFSConv2d(1, CH, kernel_size=3, stride=2, padding=1,
                             t_max=40.0, dtype=DTYPE, device=DEVICE,
                             w_scale=0.35, bias_val=0.5, grid_pts=301)
        H2 = (28 + 2 - 3) // 2 + 1  # 14
        crepr = CH * H2 * H2
        fc = ExactTTFSLinear(crepr, 10, t_max=40.0, dtype=DTYPE,
                             device=DEVICE, bias_val=1.5, grid_pts=301)
        return ConvSNN(c1, fc).to(DEVICE), latency_encode(X28, t_max=40.0), y
    if depth == 2:
        c1 = ExactTTFSConv2d(1, CH, kernel_size=3, stride=1, padding=1,
                             t_max=40.0, dtype=DTYPE, device=DEVICE,
                             w_scale=0.35, bias_val=0.5, grid_pts=301)
        c2 = ExactTTFSConv2d(CH, 2 * CH, kernel_size=3, stride=2, padding=1,
                             t_max=40.0, dtype=DTYPE, device=DEVICE,
                             w_scale=0.35, bias_val=0.5, grid_pts=301)
        H2 = (14 + 2 - 3) // 2 + 1  # 7
        crepr = 2 * CH * H2 * H2
        fc = ExactTTFSLinear(crepr, 10, t_max=40.0, dtype=DTYPE,
                             device=DEVICE, bias_val=1.5, grid_pts=301)
        return ConvStackSNN(c1, c2, fc, H2, H2).to(DEVICE), \
            latency_encode(X14, t_max=40.0), y


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    CH = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    depth = sys.argv[4] if len(sys.argv) > 4 else "1"
    depthkey = "28" if depth == "28" else int(depth)
    model, t_enc, y = build(depthkey, N, CH)
    print(f"Conv SNN depth={depth}: N={N} epochs={EPOCHS} channels={CH}", flush=True)
    train(model, t_enc, y, EPOCHS)


def train(model, t_enc, y, epochs, bs=64, lr=3e-3, t_max=40.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    N = y.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(N)
        tot, nb = 0.0, 0
        t0 = time.time()
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            xb = t_enc[idx]
            yb = y[idx]
            opt.zero_grad()
            out = model(xb)
            loss = latency_cross_entropy(out, yb, t_max)
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        acc = eval_acc(model, t_enc, y, bs=256, t_max=t_max)
        sched.step()
        print(f"  ep {ep + 1:2d}/{epochs} loss={tot / nb:.4f} acc={acc:.1%} "
              f"lr={opt.param_groups[0]['lr']:.1e} ({time.time() - t0:.1f}s)", flush=True)


def eval_acc(model, t_enc, y, bs=256, t_max=40.0):
    model.eval()
    with torch.no_grad():
        correct = total = 0
        for i in range(0, y.shape[0], bs):
            yb = y[i:i + bs]
            out = model(t_enc[i:i + bs])
            pred = out.argmin(dim=0)
            correct += int((pred == yb).sum())
            total += len(yb)
    model.train()
    return correct / max(total, 1)


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    CH = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    X, y = load_mnist(N)
    X14 = downsample(X, 2).view(N, 1, 14, 14)
    conv = ExactTTFSConv2d(1, CH, kernel_size=3, stride=1, padding=1,
                           t_max=40.0, dtype=DTYPE, device=DEVICE,
                           w_scale=0.35, bias_val=0.5, grid_pts=301)
    fc = ExactTTFSLinear(CH * 14 * 14, 10, t_max=40.0, dtype=DTYPE,
                         device=DEVICE, bias_val=1.5, grid_pts=301)
    model = ConvSNN(conv, fc).to(DEVICE)
    t_enc = latency_encode(X14, t_max=40.0)
    print(f"Conv SNN: N={N} epochs={EPOCHS} channels={CH} 14x14", flush=True)
    train(model, t_enc, y, EPOCHS)


if __name__ == "__main__":
    main()
