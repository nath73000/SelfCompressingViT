import torch
import torch.nn as nn
from selfcompress import SelfCompressingLinear
from typing import Tuple



def get_2d_sincos_pos_embed(embed_dim: int, grid_shape: tuple[int, int]) -> torch.Tensor:
    """
    embed_dim : dimension totale D (ex: 768)
    grid_shape : tuple (H, W) du nombre de patchs (ex: (14, 14))
    """
    assert embed_dim % 2 == 0, "embed_dim doit être divisible par 2"
    h, w = grid_shape

    # 1. Grille de coordonnées (H, W)
    # y va de 0 à h-1, x va de 0 à w-1
    grid_y = torch.arange(h, dtype=torch.float32)
    grid_x = torch.arange(w, dtype=torch.float32)
    
    # meshgrid génère deux matrices (H, W) de coordonnées
    grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')

    # Aplatir en vecteurs 1D de longueur H * W
    grid_y = grid_y.flatten()  # Forme: (H*W,)
    grid_x = grid_x.flatten()  # Forme: (H*W,)

    # 2. Calcul des encodages 1D pour chaque axe avec embed_dim // 2
    dim_per_axis = embed_dim // 2
    emb_y = get_1d_sincos_pos_embed(dim_per_axis, grid_y)  # Forme: (H*W, D/2)
    emb_x = get_1d_sincos_pos_embed(dim_per_axis, grid_x)  # Forme: (H*W, D/2)

    # 3. Concaténation le long de la dernière dimension
    pos_embed = torch.cat([emb_y, emb_x], dim=1)  # Forme: (H*W, D)

    return pos_embed



def get_1d_sincos_pos_embed(dim: int, pos: torch.Tensor) -> torch.Tensor:
    """
    dim : dimension de l'encodage pour cet axe (doit être paire)
    pos : tenseur 1D de coordonnées de taille (N,)
    """
    assert dim % 2 == 0
    # Calcul des fréquences : omega = 1 / (10000 ** (2i / dim))
    omega = torch.arange(dim // 2, dtype=torch.float32)
    omega /= (dim / 2)
    omega = 1.0 / (10000 ** omega)  # Forme: (dim / 2,)

    # Multiplication matricielle externe : (N, 1) * (1, dim/2) -> (N, dim/2)
    out = torch.einsum('m,d->md', pos, omega)
    
    # Sinus et Cosinus entrelacés ou concaténés
    emb_sin = torch.sin(out)  # (N, dim/2)
    emb_cos = torch.cos(out)  # (N, dim/2)

    return torch.cat([emb_sin, emb_cos], dim=1)  # (N, dim)



class PositionalEncoding2D(torch.nn.Module):
    def __init__(
            self,
            image_shape: Tuple[int, int, int],
            patch_shape: Tuple[int, int],
            embedding_size: int
    ):
        super().__init__()

        _, image_h, image_w = image_shape[0], image_shape[1], image_shape[2]
        patch_h, patch_w    = patch_shape[0], patch_shape[1]
        grid_h = image_h // patch_h
        grid_w = image_w // patch_w
        self.num_patches = grid_h * grid_w

        # Générer l'encodage fixe (pas de token CLS)
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim=embedding_size,
            grid_shape=(grid_h, grid_w)
        )

        # Enregistrer en buffer figé (non entraînable)
        # Forme finale : (1, num_patches, embed_dim)
        self.register_buffer("pos_encoding", pos_embed.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor, idx=None) -> torch.Tensor:
        pos = self.pos_encoding.expand(x.shape[0], -1, -1)
        if idx is not None:
            pos = pos[:, idx, :]
        return x + pos