"""
JaT-AudioSR Model Implementation (Just audio Transformers)
Adapted for DAC codec latent: [B, 1024, T/557]

Key differences from JiT-AudioSR:
- DAC latent: [B, 1024, T] (1D sequence, not 2D)
- Much higher channel dimension (1024 vs 16)
- 1D patching along time axis only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion timesteps"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        Args:
            t: [B] timesteps in range [0, 1]
        Returns:
            [B, dim] time embeddings
        """
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class BottleneckPatchEmbed1D(nn.Module):
    """
    1D Patch Embedding with bottleneck for DAC latents
    DAC latent: [B, C, T] where C=1024 (very high-dim)

    Patches along time axis: [B, C, T] → [B, T//patch_len, C*patch_len]
    Then compress via bottleneck
    """

    def __init__(self, patch_len, in_chans, embed_dim, bottleneck_dim):
        super().__init__()
        self.patch_len = patch_len
        self.flatten_dim = patch_len * in_chans

        # Bottleneck projection
        self.proj = nn.Sequential(
            nn.Linear(self.flatten_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, embed_dim)
        )

        print(f"[PatchEmbed1D] {self.flatten_dim}-d → {bottleneck_dim}-d → {embed_dim}-d")

    def forward(self, x):
        """
        Args:
            x: [B, C, T] latent sequence
        Returns:
            [B, N, embed_dim] patches, where N = T // patch_len
        """
        B, C, T = x.shape
        P = self.patch_len

        # Ensure T is divisible by patch_len (will pad in model if needed)
        assert T % P == 0, f"T={T} must be divisible by patch_len={P}"

        # Reshape to patches: [B, C, T] → [B, T//P, C*P]
        x = x.reshape(B, C, T // P, P)  # [B, C, N, P]
        x = x.permute(0, 2, 1, 3)  # [B, N, C, P]
        x = x.reshape(B, T // P, C * P)  # [B, N, C*P]

        # Project through bottleneck
        x = self.proj(x)  # [B, N, embed_dim]
        return x


class DiTBlock(nn.Module):
    """
    Diffusion Transformer Block with adaLN-Zero
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )

        # adaLN modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, t_emb):
        """
        Args:
            x: [B, N, D] sequence
            t_emb: [B, D] time embedding
        Returns:
            [B, N, D] output
        """
        # Get modulation parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(t_emb).chunk(6, dim=1)

        # Self-Attention with adaLN
        x_norm1 = self.norm1(x)
        x_norm1 = x_norm1 * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_output, _ = self.attn(x_norm1, x_norm1, x_norm1)
        x = x + gate_msa.unsqueeze(1) * attn_output

        # MLP with adaLN
        x_norm2 = self.norm2(x)
        x_norm2 = x_norm2 * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_output = self.mlp(x_norm2)
        x = x + gate_mlp.unsqueeze(1) * mlp_output

        return x


class JaT_AudioSR(nn.Module):
    """
    JaT architecture for Audio Super-Resolution using DAC codec
    DAC latent: [B, 1024, T/557]
    """

    def __init__(self,
                 input_channels=1024,     # DAC latent channels
                 cond_channels=1024,      # LR condition latent channels
                 patch_len=4,             # Temporal patch length
                 hidden_size=768,         # Transformer hidden size
                 depth=12,                # Number of transformer blocks
                 num_heads=12,            # Attention heads
                 bottleneck_dim=128,      # Bottleneck dimension
                 mlp_ratio=4.0):
        super().__init__()

        self.input_channels = input_channels
        self.patch_len = patch_len
        self.hidden_size = hidden_size

        print("=" * 60)
        print("Initializing JaT-AudioSR Model (DAC version)")
        print("=" * 60)
        print(f"  DAC Latent shape: [B, {input_channels}, T/557]")
        print(f"  Patch length (temporal): {patch_len}")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Depth: {depth} blocks")
        print(f"  Bottleneck dim: {bottleneck_dim}")

        # 1. Patch Embedding with bottleneck
        # Input: concat(noisy_latent, cond_latent) = 2048 channels
        total_in_channels = input_channels + cond_channels
        self.patch_embed = BottleneckPatchEmbed1D(
            patch_len=patch_len,
            in_chans=total_in_channels,
            embed_dim=hidden_size,
            bottleneck_dim=bottleneck_dim
        )

        # 2. Positional Embedding (learnable)
        # Max sequence length for 10s audio: T≈861 → patches≈215 (with patch_len=4)
        self.max_len = 2048
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_len, hidden_size))

        # 3. Time Embedding (for diffusion timestep)
        self.t_embedder = nn.Sequential(
            TimeEmbedding(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # 4. Transformer Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        ])

        # 5. Final Layer (project back to patch dimension)
        patch_out_dim = patch_len * input_channels
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
            nn.Linear(hidden_size, patch_out_dim)
        )

        self.initialize_weights()

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"  Total params: {total_params / 1e6:.2f}M")
        print("=" * 60)

    def initialize_weights(self):
        """Initialize weights following best practices"""
        # Positional embedding
        nn.init.normal_(self.pos_embed, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out final layer
        nn.init.constant_(self.final_layer[-1].weight, 0)
        nn.init.constant_(self.final_layer[-1].bias, 0)

    def unpatchify(self, x, B, C, T_orig):
        """
        Convert patches back to 1D latent sequence
        Args:
            x: [B, N, patch_out_dim] where patch_out_dim = C * patch_len
            C: number of channels
            T_orig: original temporal length
        Returns:
            [B, C, T_orig]
        """
        P = self.patch_len
        N = x.shape[1]

        # Reshape: [B, N, C*P] → [B, N, C, P] → [B, C, N, P] → [B, C, T]
        x = x.reshape(B, N, C, P)  # [B, N, C, P]
        x = x.permute(0, 2, 1, 3)  # [B, C, N, P]
        x = x.reshape(B, C, N * P)  # [B, C, T]

        # Trim to original length if padded
        if x.shape[-1] > T_orig:
            x = x[:, :, :T_orig]

        return x

    def forward(self, x_t, t, x_cond):
        """
        Forward pass
        Args:
            x_t: [B, 1024, T] noisy DAC latent
            t: [B] timesteps in range [0, 1]
            x_cond: [B, 1024, T] LR condition DAC latent
        Returns:
            x_pred: [B, 1024, T] predicted clean latent (x-prediction)
        """
        B, C, T_orig = x_t.shape

        # Pad to make T divisible by patch_len
        P = self.patch_len
        pad_len = (P - T_orig % P) % P
        if pad_len > 0:
            x_t = F.pad(x_t, (0, pad_len))
            x_cond = F.pad(x_cond, (0, pad_len))

        T = x_t.shape[-1]

        # 0. Concat conditioning along channel dimension
        x_in = torch.cat([x_t, x_cond], dim=1)  # [B, 2048, T]

        # 1. Patch Embedding
        x = self.patch_embed(x_in)  # [B, N, hidden_size]
        N = x.shape[1]

        # 2. Add Positional Embedding
        if N <= self.max_len:
            x = x + self.pos_embed[:, :N, :]
        else:
            raise ValueError(f"Sequence length {N} exceeds max_len {self.max_len}")

        # 3. Time Embedding
        t_emb = self.t_embedder(t)  # [B, hidden_size]

        # 4. Transformer Blocks
        for block in self.blocks:
            x = block(x, t_emb)

        # 5. Final Projection
        x = self.final_layer(x)  # [B, N, C*P]

        # 6. Unpatchify
        x_pred = self.unpatchify(x, B, self.input_channels, T)  # [B, C, T]

        # 7. Remove padding if added
        if pad_len > 0:
            x_pred = x_pred[:, :, :T_orig]

        return x_pred  # x-prediction: directly output clean latent


def test_model():
    """Quick test of model dimensions"""
    print("\n" + "=" * 60)
    print("Model Dimension Test")
    print("=" * 60)

    # Create model (smaller for testing)
    model = JaT_AudioSR(
        input_channels=1024,
        cond_channels=1024,
        patch_len=4,
        hidden_size=384,  # Smaller for testing
        depth=6,
        num_heads=6,
        bottleneck_dim=128
    )

    # Test forward pass
    B = 2
    T = 861  # ~10s audio with DAC
    x_t = torch.randn(B, 1024, T)
    t = torch.rand(B)
    x_cond = torch.randn(B, 1024, T)

    print(f"\nInput shapes:")
    print(f"  x_t: {list(x_t.shape)}")
    print(f"  t: {list(t.shape)}")
    print(f"  x_cond: {list(x_cond.shape)}")

    with torch.no_grad():
        output = model(x_t, t, x_cond)

    print(f"\nOutput shape: {list(output.shape)}")
    assert output.shape == x_t.shape, "Output shape mismatch!"
    print("✅ Model test passed!")

    # Test with different lengths
    print("\n" + "=" * 60)
    print("Testing variable length...")
    T2 = 433  # Different length
    x_t2 = torch.randn(B, 1024, T2)
    x_cond2 = torch.randn(B, 1024, T2)

    with torch.no_grad():
        output2 = model(x_t2, t, x_cond2)

    print(f"  Input T={T2}, Output shape: {list(output2.shape)}")
    assert output2.shape == x_t2.shape, "Variable length test failed!"
    print("✅ Variable length test passed!")


if __name__ == '__main__':
    test_model()
