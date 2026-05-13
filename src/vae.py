"""
Conditional Variational Autoencoder para estructuras quimicas.

Encoder: conv 3 bloques -> mu, logvar
Decoder: latent + class embedding -> deconv 3 bloques
Reconstruccion sobre imagenes 64x64 (las 224x224 son demasiado pesadas
para entrenar un VAE razonable en una GPU domestica en pocos minutos).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Bloque conv + BN + LeakyReLU, stride=2 para reducir resolucion a la mitad."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)), 0.2, inplace=True)


class DeconvBlock(nn.Module):
    """Bloque ConvTranspose + BN + ReLU, duplica la resolucion."""
    def __init__(self, in_ch, out_ch, last=False):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_ch) if not last else None
        self.last = last

    def forward(self, x):
        x = self.deconv(x)
        if self.last:
            return torch.sigmoid(x)
        return F.relu(self.bn(x), inplace=True)


class ConditionalVAE(nn.Module):
    """
    CVAE con embedding de clase.

    Args:
        num_classes: numero de compuestos (~196)
        latent_dim:  dimension del espacio latente z
        embed_dim:   tamaño del embedding de clase
        img_size:    asumimos cuadrada (default 64)
        base_ch:     canales del primer bloque conv
    """
    def __init__(self, num_classes, latent_dim=64, embed_dim=32,
                 img_size=64, base_ch=32):
        super().__init__()
        assert img_size % 8 == 0, "img_size debe ser multiplo de 8 (3 bloques stride-2)"
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.img_size = img_size

        self.class_emb = nn.Embedding(num_classes, embed_dim)

        # Encoder: img -> features
        self.enc1 = ConvBlock(3, base_ch)            # 64 -> 32
        self.enc2 = ConvBlock(base_ch, base_ch * 2)  # 32 -> 16
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)  # 16 -> 8

        feat_dim = (base_ch * 4) * (img_size // 8) * (img_size // 8)
        self.fc_mu = nn.Linear(feat_dim + embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(feat_dim + embed_dim, latent_dim)

        # Decoder: z + class -> img
        self.fc_dec = nn.Linear(latent_dim + embed_dim, feat_dim)
        self.dec_base_ch = base_ch * 4
        self.dec_spatial = img_size // 8
        self.dec1 = DeconvBlock(base_ch * 4, base_ch * 2)   # 8 -> 16
        self.dec2 = DeconvBlock(base_ch * 2, base_ch)       # 16 -> 32
        self.dec3 = DeconvBlock(base_ch, 3, last=True)      # 32 -> 64

    def encode(self, x, y):
        h = self.enc1(x)
        h = self.enc2(h)
        h = self.enc3(h)
        h = h.flatten(1)
        c = self.class_emb(y)
        h = torch.cat([h, c], dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        c = self.class_emb(y)
        h = torch.cat([z, c], dim=1)
        h = self.fc_dec(h)
        h = h.view(-1, self.dec_base_ch, self.dec_spatial, self.dec_spatial)
        h = self.dec1(h)
        h = self.dec2(h)
        return self.dec3(h)

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, y)
        return recon, mu, logvar

    @torch.no_grad()
    def sample(self, y, n=1, device=None):
        """Genera n imagenes condicionadas a la clase y."""
        device = device or next(self.parameters()).device
        if not torch.is_tensor(y):
            y = torch.tensor([y] * n, dtype=torch.long, device=device)
        elif y.numel() == 1:
            y = y.repeat(n).to(device)
        z = torch.randn(y.size(0), self.latent_dim, device=device)
        return self.decode(z, y)


def vae_loss(recon, x, mu, logvar, beta=1.0):
    """
    Reconstruction (MSE) + beta * KL divergence.

    Reduccion 'sum' para evitar dependencia del batch_size en la magnitud.
    Se divide por batch_size al final para que las metricas sean comparables.
    """
    bs = x.size(0)
    recon_loss = F.mse_loss(recon, x, reduction='sum') / bs
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / bs
    return recon_loss + beta * kl, recon_loss, kl
