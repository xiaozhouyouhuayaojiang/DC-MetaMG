import torch
import torch.nn as nn
import numpy as np


# ─────────────────────────────────────────────
# Laplacian Positional Encoding
# ─────────────────────────────────────────────
def laplacian_pe(adj: np.ndarray, k: int) -> torch.Tensor:
    """
    Compute normalized Laplacian positional encoding.
    adj: (N, N) adjacency matrix (numpy)
    k:   number of eigenvectors to keep
    Returns: (N, k) tensor
    """
    N = adj.shape[0]
    deg = adj.sum(axis=1)
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    L = np.eye(N) - D_inv_sqrt @ adj @ D_inv_sqrt   # normalized Laplacian
    eigvals, eigvecs = np.linalg.eigh(L)             # eigvals sorted ascending
    pe = eigvecs[:, :k]                              # smallest-k eigenvectors
    return torch.FloatTensor(pe)


# ─────────────────────────────────────────────
# FAT Module  (Transformer-based alignment)
# ─────────────────────────────────────────────
class FATModule(nn.Module):
    """
    Feature Alignment based on Transformer.
    Aligns miRNA k-mer / base-sequence features with drug bond-angle features.
    """
    def __init__(self, dim_mirna: int, dim_drug: int, d_model: int = 128,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1,
                 pe_k: int = 16):
        super().__init__()
        self.pe_k = pe_k

        # Project each modality + positional encoding into d_model
        self.proj_mirna = nn.Linear(dim_mirna + pe_k, d_model)
        self.proj_drug  = nn.Linear(dim_drug  + pe_k, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self,  x_drug: torch.Tensor, x_mirna: torch.Tensor,
               ) -> torch.Tensor:
        """
        x_mirna : (B, dim_mirna)
        x_drug  : (B, dim_drug)
        pe_*    : (B, pe_k)  – Laplacian PE for each sample
        """
        # Append positional encoding


        # Stack as sequence of length 2 for the transformer
        seq = torch.stack([x_drug, x_mirna], dim=1)          # (B, 2, d_model)
        out = self.transformer(seq)               # (B, 2, d_model)

        # Concatenate both positions and project
        fused = self.out_proj(out.reshape(out.size(0), -1))  # (B, d_model)
        return self.norm(fused)


# ─────────────────────────────────────────────
# FAV Module  (Supervised VAE-based alignment)
# ─────────────────────────────────────────────
class FAVModule(nn.Module):
    """
    Feature Alignment based on supervised VAE.
    Aligns miRNA functional-similarity features with drug biological features.
    """
    def __init__(self, dim_mirna: int, dim_drug: int,
                 latent_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        in_dim = dim_mirna + dim_drug

        # Encoder: input → μ, log σ²
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: z → reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )

        # Supervised head: z → prediction score (used as intermediate feature)
        self.sup_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x_drug: torch.Tensor, x_mirna: torch.Tensor):
        """
        Returns: (fused, recon, mu, logvar)
        """
        x = torch.cat([x_drug, x_mirna], dim=-1)
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        fused = self.sup_head(z)
        return fused, recon, mu, logvar


# ─────────────────────────────────────────────
# MLP Classifier  (unchanged role, generalized)
# ─────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, num_in: int, num_hid1: int, num_out: int,
                 dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_in, num_hid1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_hid1, num_out),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# DCMFA  – full dual-channel model
# ─────────────────────────────────────────────
class DCMFA(nn.Module):
    """
    Dual-Channel miRNA-drug resistance prediction with Multimodal Feature Alignment.

    Channel 1 (FAT): miRNA k-mer sequence  ↔  drug bond angles
    Channel 2 (FAV): miRNA func. similarity ↔  drug biological features
    """
    def __init__(self,
                 # Channel-1 feature dims
                 dim_mirna_seq:  int,
                 dim_drug_angle: int,
                 # Channel-2 feature dims
                 dim_mirna_sim:  int,
                 dim_drug_bio:   int,
                 # FAT hyper-params
                 fat_d_model:  int = 128,
                 fat_n_heads:  int = 4,
                 fat_n_layers: int = 2,
                 fat_pe_k:     int = 16,
                 # FAV hyper-params
                 fav_latent:   int = 64,
                 fav_hidden:   int = 128,
                 # Classifier
                 mlp_hid1: int = 256,
                 mlp_hid2: int = 64,
                 dropout:  float = 0.5):
        super().__init__()

        self.fat = FATModule(
            dim_mirna=dim_mirna_seq,
            dim_drug=dim_drug_angle,
            d_model=fat_d_model,
            n_heads=fat_n_heads,
            n_layers=fat_n_layers,
            dropout=dropout,
            pe_k=fat_pe_k,
        )

        self.fav = FAVModule(
            dim_mirna=dim_mirna_sim,
            dim_drug=dim_drug_bio,
            latent_dim=fav_latent,
            hidden_dim=fav_hidden,
        )

        # Align FAT output to fav_latent dim
        self.fat_align = nn.Sequential(
            nn.Linear(fat_d_model, fav_latent),
            nn.ReLU(),
        )

        # Final MLP: concat both channels → prediction
        self.classifier = MLP(
            num_in=fav_latent * 2,
            num_hid1=mlp_hid1,
            num_out=1,
            dropout=dropout,
        )

    def forward(self,
                x_drug_angle: torch.Tensor,
                x_mirna_seq: torch.Tensor,

                x_drug_bio:   torch.Tensor,
                x_mirna_sim:  torch.Tensor):
        """
        All inputs: (B, dim)
        Returns: (pred, recon, mu, logvar)
          pred   : (B, 1)  sigmoid prediction
          recon  : (B, dim_mirna_sim + dim_drug_bio)  VAE reconstruction
          mu     : (B, fav_latent)
          logvar : (B, fav_latent)
        """
        # Channel 1
        fat_out = self.fat(x_drug_angle, x_mirna_seq)
        fat_out = self.fat_align(fat_out)                       # (B, fav_latent)

        # Channel 2
        fav_out, recon, mu, logvar = self.fav(x_drug_bio,x_mirna_sim)

        # Fuse channels and classify
        fused = torch.cat([fat_out, fav_out], dim=-1)          # (B, 2*fav_latent)
        pred  = self.classifier(fused)

        return pred, recon, mu, logvar


# ─────────────────────────────────────────────
# Backward-compat alias (old code used `mlp`)
# ─────────────────────────────────────────────
class mlp(MLP):
    def __init__(self, num_in, num_hid1, num_hid2, num_out):
        super().__init__(num_in, num_hid1, num_hid2, num_out)
