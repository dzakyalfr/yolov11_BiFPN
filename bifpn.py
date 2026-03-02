"""
BiFPN adapted for YOLOv11 (P3/P4/P5 only)

Changes from the original EfficientDet BiFPN:
- Operates only on [P3, P4, P5] — P6/P7 removed entirely.
- Channel projection layers accept arbitrary backbone channel counts.
- Learnable fast-normalized weighted fusion (w1/w2) per EfficientDet paper, 
  now reduced to the 3-level case.
- Resize ops use F.interpolate for upsampling and F.avg_pool2d for
  downsampling — both size-agnostic (no hard-coded scale factors).
- The top-level BiFPN wrapper accepts a list of (in_channels, out_channels)
  pairs for each scale so it is fully portable across YOLO model variants.
- Stacks multiple BiFPNBlock iterations (num_layers) for richer fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DepthwiseSeparableConv(nn.Module):
    """Depthwise-separable conv  →  BN  →  ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.dw = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, padding=1, groups=in_channels, bias=False,
        )
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


# ---------------------------------------------------------------------------
# Single BiFPN iteration  (3-level: P3 / P4 / P5)
# ---------------------------------------------------------------------------

class BiFPNBlock(nn.Module):
    """
    One bi-directional feature-pyramid pass over three feature levels.

    Top-down path (high-res → low-res  is skipped; we go low-res → high-res):
        P5_td  = P5                               (passthrough)
        P4_td  = w·P4  +  w·upsample(P5_td)
        P3_td  = w·P3  +  w·upsample(P4_td)

    Bottom-up path (high-res → low-res):
        P3_out = P3_td                            (passthrough)
        P4_out = w·P4  +  w·P4_td  +  w·downsample(P3_out)
        P5_out = w·P5  +  w·P5_td  +  w·downsample(P4_out)

    All fusion weights use the fast-normalized attention of EfficientDet:
        weight_i = ReLU(w_i) / (sum_j ReLU(w_j) + ε)

    Args:
        feature_size: Number of channels used throughout (same for all levels).
        epsilon: Small constant to avoid division by zero in weight fusion.
    """

    def __init__(self, feature_size: int, epsilon: float = 1e-4):
        super().__init__()
        self.epsilon = epsilon

        # --- top-down fusion convs (2 inputs: current + upsampled above) ---
        self.p4_td_conv = DepthwiseSeparableConv(feature_size, feature_size)
        self.p3_td_conv = DepthwiseSeparableConv(feature_size, feature_size)

        # --- bottom-up fusion convs (3 inputs: orig + td + downsampled below) ---
        self.p4_out_conv = DepthwiseSeparableConv(feature_size, feature_size)
        self.p5_out_conv = DepthwiseSeparableConv(feature_size, feature_size)

        # Learnable weights — shape (num_inputs, num_levels_using_that_size)
        # Top-down: 2 weights × 2 levels  (P4_td, P3_td)
        self.w_td = nn.Parameter(torch.ones(2, 2))   # [input_idx, level_idx]
        # Bottom-up: 3 weights × 2 levels  (P4_out, P5_out)
        self.w_bu = nn.Parameter(torch.ones(3, 2))   # [input_idx, level_idx]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _upsample_to(src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Upsample `src` to match the spatial size of `target`."""
        return F.interpolate(src, size=target.shape[-2:], mode="nearest")

    @staticmethod
    def _downsample_to(src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Downsample `src` to match the spatial size of `target` with avg-pool."""
        return F.adaptive_avg_pool2d(src, output_size=target.shape[-2:])

    def _normalize_w_td(self) -> torch.Tensor:
        w = F.relu(self.w_td)
        return w / (w.sum(dim=0, keepdim=True) + self.epsilon)   # (2, 2)

    def _normalize_w_bu(self) -> torch.Tensor:
        w = F.relu(self.w_bu)
        return w / (w.sum(dim=0, keepdim=True) + self.epsilon)   # (3, 2)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, features: list) -> list:
        """
        Args:
            features: [p3, p4, p5] — each a (B, feature_size, H, W) tensor.

        Returns:
            [p3_out, p4_out, p5_out] — same shapes as input.
        """
        p3, p4, p5 = features

        wt = self._normalize_w_td()    # (2, 2)  dim-0 = input, dim-1 = level
        wb = self._normalize_w_bu()    # (3, 2)

        # ── Top-down pathway ────────────────────────────────────────────────
        # P5_td is a simple passthrough (no fusion needed at the top level)
        p5_td = p5

        # P4_td fuses original P4 with upsampled P5_td
        p4_td = self.p4_td_conv(
            wt[0, 0] * p4 + wt[1, 0] * self._upsample_to(p5_td, p4)
        )

        # P3_td fuses original P3 with upsampled P4_td
        p3_td = self.p3_td_conv(
            wt[0, 1] * p3 + wt[1, 1] * self._upsample_to(p4_td, p3)
        )

        # ── Bottom-up pathway ───────────────────────────────────────────────
        # P3_out is a simple passthrough (no fusion needed at the bottom level)
        p3_out = p3_td

        # P4_out fuses original P4, P4 from top-down, and downsampled P3_out
        p4_out = self.p4_out_conv(
            wb[0, 0] * p4 + wb[1, 0] * p4_td + wb[2, 0] * self._downsample_to(p3_out, p4)
        )

        # P5_out fuses original P5, P5 from top-down, and downsampled P4_out
        p5_out = self.p5_out_conv(
            wb[0, 1] * p5 + wb[1, 1] * p5_td + wb[2, 1] * self._downsample_to(p4_out, p5)
        )

        return [p3_out, p4_out, p5_out]


# ---------------------------------------------------------------------------
# Top-level BiFPN: channel projection + stacked BiFPNBlock iterations
# ---------------------------------------------------------------------------

class BiFPN(nn.Module):
    """
    BiFPN neck for YOLOv11 — P3/P4/P5 only.

    Accepts backbone feature maps with *arbitrary* channel sizes and first
    projects each to a common `feature_size`, then runs `num_layers` rounds
    of bidirectional top-down / bottom-up fusion.

    Args:
        in_channels: Tuple/list of three ints — channels of (P3, P4, P5) from
                     the backbone.  Example: (256, 512, 1024) for YOLOv11n.
        feature_size: Inner channel width used throughout the BiFPN.
                      128 or 256 are typical choices.
        num_layers:   How many BiFPNBlock rounds to stack (default 2, original
                      EfficientDet uses 3–7 depending on compound coefficient).
        epsilon:      Denominator stabiliser for weighted fusion.

    Forward input:
        A *list* of three tensors [p3, p4, p5] produced by the backbone.

    Forward output:
        A *list* of three tensors [p3, p4, p5] ready for the Detect head.
    """

    def __init__(
        self,
        in_channels: tuple,  # (C_p3, C_p4, C_p5)
        feature_size: int = 128,
        num_layers: int = 2,
        epsilon: float = 1e-4,
    ):
        super().__init__()

        assert len(in_channels) == 3, "BiFPN expects exactly 3 input feature sizes."

        # 1×1 channel projection: backbone channels → unified feature_size
        self.proj_p3 = nn.Sequential(
            nn.Conv2d(in_channels[0], feature_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_size, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
        )
        self.proj_p4 = nn.Sequential(
            nn.Conv2d(in_channels[1], feature_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_size, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
        )
        self.proj_p5 = nn.Sequential(
            nn.Conv2d(in_channels[2], feature_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_size, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
        )

        # Stacked BiFPN passes
        self.bifpn = nn.Sequential(
            *[BiFPNBlock(feature_size, epsilon) for _ in range(num_layers)]
        )

    def forward(self, features: list) -> list:
        """
        Args:
            features: [p3, p4, p5] from the backbone.

        Returns:
            [p3_out, p4_out, p5_out] fused feature maps, all with `feature_size`
            channels and the same spatial dimensions as the respective inputs.
        """
        p3, p4, p5 = features

        # Project to uniform channel width
        p3 = self.proj_p3(p3)
        p4 = self.proj_p4(p4)
        p5 = self.proj_p5(p5)

        # Run bi-directional fusion
        return self.bifpn([p3, p4, p5])
