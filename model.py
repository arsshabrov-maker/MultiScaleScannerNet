import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_div(numerator, denominator, eps=0.05):
  sign = torch.where(denominator >= 0, 1.0, -1.0)
  return numerator / (denominator + sign * eps)


class ParallelDivMulChainLayer(nn.Module):
  """16 parallel division-multiplication paths capturing physical ratio invariants."""

  def __init__(self, in_features=16, n_paths=16):
    super().__init__()
    self.weights = nn.Parameter(torch.randn(n_paths, in_features) * 0.3 + 0.5)
    self.biases = nn.Parameter(torch.zeros(n_paths))
    self.act = nn.Tanh()

  def forward(self, x):
    u = x.unsqueeze(1) * self.weights.unsqueeze(0)
    ratio = self.act(safe_div(u[:, :, 0], u[:, :, 1]))
    prod = ratio * u[:, :, 2]
    return self.act(prod + self.biases.unsqueeze(0))


class MultiScaleScannerLayer(nn.Module):
  """
    Hierarchical Multi-Scale Layer (total width = n_groups * group_size = 100):
      1. Global Aggregator: reads all 100 units -> 16-dim global summary
      2. Global 1->1 Scanner: single shared 1->1 micro-net sweeping across all 100 units
      3. Group Aggregators: `n_groups` independent nets (group_size -> 1)
      4. Group 1->1 Scanners: `n_groups` independent 1->1 micro-nets sweeping inside each group
      5. Cross-Group Observers: `group_size` nets reading strided units across distant groups
    """

  def __init__(
      self,
      n_groups=5,
      group_size=20,
      use_cross_groups=True,
      scan_hidden=6,
  ):
    super().__init__()
    assert n_groups * group_size == 100, 'Total layer width must be 100'
    self.n_groups = n_groups
    self.group_size = group_size
    self.use_cross_groups = use_cross_groups

    # 1. Global Aggregator (100 -> 16)
    self.global_agg = nn.Sequential(nn.Linear(100, 16), nn.GELU())

    # 2. Global 1->1 Scanner running across all 100 neurons
    self.global_scanner = nn.Sequential(
        nn.Linear(1, scan_hidden), nn.GELU(), nn.Linear(scan_hidden, 1)
    )

    # 3. `n_groups` Group Aggregators (each: group_size -> 4 -> 1)
    self.grp_agg_w1 = nn.Parameter(
        torch.randn(n_groups, group_size, 4) * 0.25
    )
    self.grp_agg_b1 = nn.Parameter(torch.zeros(n_groups, 1, 4))
    self.grp_agg_w2 = nn.Parameter(torch.randn(n_groups, 4, 1) * 0.25)
    self.grp_agg_b2 = nn.Parameter(torch.zeros(n_groups, 1, 1))

    # 4. `n_groups` Group 1->1 Scanners (each runs across its `group_size` neurons)
    self.grp_scan_w1 = nn.Parameter(
        torch.randn(n_groups, 1, scan_hidden) * 0.35
    )
    self.grp_scan_b1 = nn.Parameter(torch.zeros(n_groups, 1, scan_hidden))
    self.grp_scan_w2 = nn.Parameter(
        torch.randn(n_groups, scan_hidden, 1) * 0.35
    )
    self.grp_scan_b2 = nn.Parameter(torch.zeros(n_groups, 1, 1))

    fused_dim = 16 + 100 + n_groups + 100

    # 5. Cross-Group Observers (strided neurons across distant groups)
    if self.use_cross_groups:
      self.cross_w1 = nn.Parameter(
          torch.randn(group_size, n_groups, 4) * 0.25
      )
      self.cross_b1 = nn.Parameter(torch.zeros(group_size, 1, 4))
      self.cross_w2 = nn.Parameter(torch.randn(group_size, 4, 1) * 0.25)
      self.cross_b2 = nn.Parameter(torch.zeros(group_size, 1, 1))
      fused_dim += group_size

    # Layer Arbiter combining all multi-scale streams back into 100 units
    self.arbiter = nn.Sequential(
        nn.Linear(fused_dim, 100),
        nn.LayerNorm(100),
        nn.GELU(),
    )

  def forward(self, x):
    B = x.size(0)
    x_grp = x.view(B, self.n_groups, self.group_size)

    # Stream 1: Global Aggregator
    f_glob_agg = self.global_agg(x)

    # Stream 2: Global 1->1 Scanner across all 100 neurons
    f_glob_scan = self.global_scanner(x.unsqueeze(-1)).squeeze(-1)

    # Stream 3: Group Aggregators (G networks, S -> 1)
    xg_t = x_grp.permute(1, 0, 2)
    h_ga = F.gelu(torch.bmm(xg_t, self.grp_agg_w1) + self.grp_agg_b1)
    f_grp_agg = (
        (torch.bmm(h_ga, self.grp_agg_w2) + self.grp_agg_b2)
        .squeeze(-1)
        .permute(1, 0)
    )

    # Stream 4: Group 1->1 Scanners (G networks, 1 -> 1 across S units)
    xg_scan = xg_t.reshape(self.n_groups, B * self.group_size, 1)
    h_gs = F.gelu(torch.bmm(xg_scan, self.grp_scan_w1) + self.grp_scan_b1)
    f_grp_scan = (
        (torch.bmm(h_gs, self.grp_scan_w2) + self.grp_scan_b2)
        .reshape(self.n_groups, B, self.group_size)
        .permute(1, 0, 2)
        .reshape(B, 100)
    )

    streams = [f_glob_agg, f_glob_scan, f_grp_agg, f_grp_scan]

    # Stream 5: Cross-Group Observers
    if self.use_cross_groups:
      xc_t = x_grp.permute(2, 0, 1)
      h_cr = F.gelu(torch.bmm(xc_t, self.cross_w1) + self.cross_b1)
      f_cross = (
          (torch.bmm(h_cr, self.cross_w2) + self.cross_b2)
          .squeeze(-1)
          .permute(1, 0)
      )
      streams.append(f_cross)

    combined = torch.cat(streams, dim=1)
    return x + self.arbiter(combined)


class DeepMultiScaleNet(nn.Module):
  """Stacked MultiScaleScannerLayers + Exit Mini-Network."""

  def __init__(
      self,
      in_dim,
      n_layers=2,
      n_groups=5,
      group_size=20,
      use_cross_groups=True,
      out_dim=128,
      normalize_output=True,
  ):
    super().__init__()
    self.normalize_output = normalize_output
    self.input_proj = nn.Sequential(
        nn.Linear(in_dim, 100),
        nn.LayerNorm(100),
        nn.GELU(),
    )
    self.blocks = nn.ModuleList([
        MultiScaleScannerLayer(
            n_groups=n_groups,
            group_size=group_size,
            use_cross_groups=use_cross_groups,
            scan_hidden=6,
        )
        for _ in range(n_layers)
    ])
    self.exit_mini_net = nn.Sequential(
        nn.Linear(100, 64),
        nn.GELU(),
        nn.Linear(64, out_dim),
    )

  def forward(self, x):
    h = self.input_proj(x)
    for block in self.blocks:
      h = block(h)
    out = self.exit_mini_net(h)
    if self.normalize_output:
      return F.normalize(out, p=2, dim=-1)
    return out


class HybridMultiScaleDivMulNet(nn.Module):
  """
    Competition-winning variant combining 16 Parallel Div/Mul Rational Chains
    with 2 Stacked MultiScaleScanner Layers (5x20 + Cross-Groups) + Exit Mini-Net.
    """

  def __init__(self, in_dim=109, emb_dim=128):
    super().__init__()
    self.front = nn.Sequential(nn.Linear(in_dim, 16), nn.Tanh())
    self.div_mul = ParallelDivMulChainLayer(in_features=16, n_paths=16)
    self.input_proj = nn.Sequential(
        nn.Linear(in_dim + 16, 100),
        nn.LayerNorm(100),
        nn.GELU(),
    )
    self.block1 = MultiScaleScannerLayer(
        n_groups=5, group_size=20, use_cross_groups=True
    )
    self.block2 = MultiScaleScannerLayer(
        n_groups=5, group_size=20, use_cross_groups=True
    )
    self.exit_net = nn.Sequential(
        nn.Linear(100, 64),
        nn.GELU(),
        nn.Linear(64, emb_dim),
    )

  def forward(self, x):
    h_rat = self.div_mul(self.front(x))
    h = self.input_proj(torch.cat([x, h_rat], dim=1))
    h = self.block1(h)
    h = self.block2(h)
    return F.normalize(self.exit_net(h), p=2, dim=-1)
