import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model import DeepMultiScaleNet


def generate_multiscale_pairs(n_samples=5000, dim=64, seed=42):
  rng = np.random.RandomState(seed)
  base_signals = rng.randn(n_samples, dim).astype(np.float32)

  for i in range(0, dim, 8):
    base_signals[:, i + 1] = np.sin(base_signals[:, i] * 2.0)
  base_signals[:, -1] = base_signals[:, 0] * base_signals[:, 32]

  view_a = base_signals + rng.randn(n_samples, dim).astype(np.float32) * 0.25
  view_b = base_signals + rng.randn(n_samples, dim).astype(np.float32) * 0.25
  return view_a, view_b


def evaluate_retrieval(model, xa, xp):
  model.eval()
  with torch.no_grad():
    ea = model(torch.tensor(xa))
    ep = model(torch.tensor(xp))
    sim = ea @ ep.T
    ranks = torch.argsort(sim, dim=1, descending=True)
    targets = torch.arange(len(xa)).unsqueeze(1)
    top1 = (ranks[:, :1] == targets).any(dim=1).float().mean().item() * 100.0
    top10 = (ranks[:, :10] == targets).any(dim=1).float().mean().item() * 100.0
  return top1, top10


def main():
  Xa, Xp = generate_multiscale_pairs(n_samples=5000, dim=64, seed=42)
  Xa_tr, Xp_tr = Xa[:4000], Xp[:4000]
  Xa_va, Xp_va = Xa[4000:], Xp[4000:]

  loader = DataLoader(
      TensorDataset(torch.tensor(Xa_tr), torch.tensor(Xp_tr)),
      batch_size=256,
      shuffle=True,
  )

  configs = {
      '1. Base (10x10, No Cross-Groups)': dict(
          n_groups=10, group_size=10, use_cross_groups=False
      ),
      '2. Standard + Cross-Groups (10x10 + Cross)': dict(
          n_groups=10, group_size=10, use_cross_groups=True
      ),
      '3. Smaller Groups + Cross (20x5 + Cross)': dict(
          n_groups=20, group_size=5, use_cross_groups=True
      ),
      '4. Larger Groups + Cross (5x20 + Cross)': dict(
          n_groups=5, group_size=20, use_cross_groups=True
      ),
  }

  print('=' * 80)
  print('MULTI-SCALE SCANNER NET: ABLATION BENCHMARK (1,000 Unseen Pairs)')
  print('=' * 80)

  for name, kwargs in configs.items():
    torch.manual_seed(42)
    model = DeepMultiScaleNet(in_dim=64, n_layers=2, out_dim=64, **kwargs)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

    for _ in range(15):
      model.train()
      for ba, bp in loader:
        opt.zero_grad()
        ea, ep = model(ba), model(bp)
        logits = (ea @ ep.T) * 15.0
        targets = torch.arange(len(ba))
        loss = (
            F.cross_entropy(logits, targets)
            + F.cross_entropy(logits.T, targets)
        ) * 0.5
        loss.backward()
        opt.step()

    t1, t10 = evaluate_retrieval(model, Xa_va, Xp_va)
    print(
        f'{name:<44} | Params: {params:<6} | Top-1: {t1:5.2f}% | Top-10:'
        f' {t10:5.2f}%'
    )
  print('=' * 80)


if __name__ == '__main__':
  main()
