from __future__ import annotations

import torch
import torch.nn as nn

# Paper mapping (Daudt et al., ICIP 2018):
#   FC-EF        -> UNet           (early fusion: concat the two RGB images)
#   FC-Siam-diff -> SiameseUNet(fusion="diff")
#   FC-Siam-conc -> SiameseUNet(fusion="conc")
# We intentionally do NOT implement STANet attention (Chen & Shi, 2020).


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    """FC-EF style six-channel U-Net: RGB(t0)+RGB(t1) -> change logits."""

    def __init__(self, in_channels: int = 6, base: int = 16) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.bottleneck = ConvBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.enc1(x)
        b = self.enc2(self.pool(a))
        c = self.enc3(self.pool(b))
        z = self.bottleneck(self.pool(c))
        z = self.dec3(torch.cat([self.up3(z), c], dim=1))
        z = self.dec2(torch.cat([self.up2(z), b], dim=1))
        z = self.dec1(torch.cat([self.up1(z), a], dim=1))
        return self.out(z)


class Encoder(nn.Module):
    """Shared Siamese encoder that returns multi-scale features."""

    def __init__(self, in_channels: int = 3, base: int = 16) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.bottleneck = ConvBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        a = self.enc1(x)
        b = self.enc2(self.pool(a))
        c = self.enc3(self.pool(b))
        z = self.bottleneck(self.pool(c))
        return a, b, c, z


class SiameseUNet(nn.Module):
    """Shared-weight Siamese U-Net (Daudt FC-Siam-diff / FC-Siam-conc).

    Both dates share one encoder. Skip/bottleneck features are fused either by
    absolute difference (`diff`) or by channel concatenation (`conc`), then
    decoded into a change logit map.
    """

    def __init__(self, in_channels: int = 3, base: int = 16, fusion: str = "diff") -> None:
        super().__init__()
        if fusion not in {"diff", "conc"}:
            raise ValueError(f"fusion must be 'diff' or 'conc', got {fusion!r}")
        self.fusion = fusion
        self.encoder = Encoder(in_channels=in_channels, base=base)
        skip_mult = 1 if fusion == "diff" else 2

        # Bottleneck fused width = skip_mult * base * 8
        self.up3 = nn.ConvTranspose2d(skip_mult * base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base * 4 + skip_mult * base * 4, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base * 2 + skip_mult * base * 2, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base + skip_mult * base, base)
        self.out = nn.Conv2d(base, 1, kernel_size=1)

    def _fuse(self, fa: torch.Tensor, fb: torch.Tensor) -> torch.Tensor:
        if self.fusion == "diff":
            return torch.abs(fa - fb)
        return torch.cat([fa, fb], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Common dataloader interface: 6-channel concat(A,B); split inside the net.
        if x.shape[1] != 6:
            raise ValueError(f"SiameseUNet expects 6-channel input, got {tuple(x.shape)}")
        a0, a1 = x[:, :3], x[:, 3:]
        s1a, s2a, s3a, za = self.encoder(a0)
        s1b, s2b, s3b, zb = self.encoder(a1)
        d1 = self._fuse(s1a, s1b)
        d2 = self._fuse(s2a, s2b)
        d3 = self._fuse(s3a, s3b)
        dz = self._fuse(za, zb)
        z = self.dec3(torch.cat([self.up3(dz), d3], dim=1))
        z = self.dec2(torch.cat([self.up2(z), d2], dim=1))
        z = self.dec1(torch.cat([self.up1(z), d1], dim=1))
        return self.out(z)


MODEL_ALIASES = {
    "unet": ("unet", None),
    "fc_ef": ("unet", None),
    "siamese": ("siamese", "diff"),
    "siamese_diff": ("siamese", "diff"),
    "fc_siam_diff": ("siamese", "diff"),
    "siamese_conc": ("siamese", "conc"),
    "fc_siam_conc": ("siamese", "conc"),
}


def build_model(name: str, base_channels: int = 16) -> nn.Module:
    key = name.lower().strip()
    if key not in MODEL_ALIASES:
        raise ValueError(
            f"Unknown model: {name}. Expected one of {sorted(MODEL_ALIASES)}"
        )
    family, fusion = MODEL_ALIASES[key]
    if family == "unet":
        return UNet(in_channels=6, base=base_channels)
    return SiameseUNet(in_channels=3, base=base_channels, fusion=fusion or "diff")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def paper_name(model_key: str) -> str:
    """Human-readable Daudt-style label for reports/tables."""
    key = model_key.lower().strip()
    mapping = {
        "unet": "FC-EF (6-channel U-Net)",
        "fc_ef": "FC-EF (6-channel U-Net)",
        "siamese": "FC-Siam-diff",
        "siamese_diff": "FC-Siam-diff",
        "fc_siam_diff": "FC-Siam-diff",
        "siamese_conc": "FC-Siam-conc",
        "fc_siam_conc": "FC-Siam-conc",
    }
    return mapping.get(key, model_key)
