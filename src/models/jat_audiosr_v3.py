"""
JaT-AudioSR Model V3 (Modern Architecture)
V3 Improvements over V2:
1. ✅ RMSNorm (替代LayerNorm) - 更快、更稳定
2. ✅ Grouped-Query Attention (GQA) for efficiency
3. ✅ RoPE (Rotary Position Embeddings) for better extrapolation
4. ✅ DropPath (Stochastic Depth) for regularization
5. ✅ Deeper and wider architecture for better quality
6. ✅ U-shaped timestep sampling for flow matching

Modern LLM Stack (LLaMA 3, Mistral风格):
- RMSNorm代替LayerNorm
- GELU激活函数 (SwiGLU留待未来)
- GQA注意力机制
- RoPE位置编码

References:
- DiT: https://arxiv.org/abs/2212.09748
- GQA: https://arxiv.org/abs/2305.13245
- RMSNorm: https://arxiv.org/abs/1910.07467
- Improved Rectified Flow: https://arxiv.org/abs/2405.20320
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# V3改进：使用PyTorch 2.8+官方RMSNorm实现
# 参考：https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html
# 优势：
# - 官方实现，性能优化更好
# - 与LLaMA 3, Mistral等现代LLM保持一致
# - 支持fused kernel加速（CUDA优化）


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class RoPE(nn.Module):
    """Rotary Position Embedding (RoPE)"""

    def __init__(self, dim, max_seq_len=4096, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Precompute cos and sin
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)  # [max_seq_len, dim//2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [max_seq_len, dim]
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def forward(self, x, seq_len):
        """
        Args:
            x: [B, N, H, D] where N is sequence length, H is num heads, D is head dim
            seq_len: actual sequence length
        Returns:
            Rotated x with same shape
        """
        # Get cached cos/sin up to seq_len
        cos = self.cos_cached[:seq_len, :]  # [N, D]
        sin = self.sin_cached[:seq_len, :]  # [N, D]

        # Apply rotation
        x_rot = self.rotate_half(x)
        x = x * cos.unsqueeze(0).unsqueeze(2) + x_rot * sin.unsqueeze(0).unsqueeze(2)
        return x

    @staticmethod
    def rotate_half(x):
        """Rotate half the hidden dims of the input"""
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA)
    Groups multiple query heads to share KV heads
    """

    def __init__(self, hidden_size, num_q_heads, num_kv_heads, dropout=0.0):
        super().__init__()
        assert hidden_size % num_q_heads == 0, "hidden_size must be divisible by num_q_heads"
        assert num_q_heads % num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads"

        self.hidden_size = hidden_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_q_heads // num_kv_heads
        self.head_dim = hidden_size // num_q_heads

        # Q projection (all query heads)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # K, V projections (fewer heads)
        kv_hidden_size = self.num_kv_heads * self.head_dim
        self.k_proj = nn.Linear(hidden_size, kv_hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, kv_hidden_size, bias=False)

        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout)

        # RoPE for Q and K
        self.rope = RoPE(self.head_dim)

    def forward(self, x):
        """
        Args:
            x: [B, N, D] input sequence
        Returns:
            [B, N, D] output
        """
        B, N, D = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x).view(B, N, self.num_q_heads, self.head_dim)  # [B, N, num_q_heads, head_dim]
        K = self.k_proj(x).view(B, N, self.num_kv_heads, self.head_dim)  # [B, N, num_kv_heads, head_dim]
        V = self.v_proj(x).view(B, N, self.num_kv_heads, self.head_dim)  # [B, N, num_kv_heads, head_dim]

        # Apply RoPE to Q and K
        Q = self.rope(Q, N)
        K = self.rope(K, N)

        # Expand K, V to match number of query heads
        # Each KV head is shared by num_groups query heads
        K = K.repeat_interleave(self.num_groups, dim=2)  # [B, N, num_q_heads, head_dim]
        V = V.repeat_interleave(self.num_groups, dim=2)  # [B, N, num_q_heads, head_dim]

        # Transpose for attention: [B, num_heads, N, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # [B, num_heads, N, head_dim]

        # Concatenate heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, D)
        output = self.out_proj(attn_output)

        return output


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

        assert T % P == 0, f"T={T} must be divisible by patch_len={P}"

        # Reshape to patches: [B, C, T] → [B, T//P, C*P]
        x = x.reshape(B, C, T // P, P)  # [B, C, N, P]
        x = x.permute(0, 2, 1, 3)  # [B, N, C, P]
        x = x.reshape(B, T // P, C * P)  # [B, N, C*P]

        # Project through bottleneck
        x = self.proj(x)  # [B, N, embed_dim]
        return x


class DiTBlock_GQA(nn.Module):
    """
    Diffusion Transformer Block with GQA and adaLN-Zero
    V3: 使用RMSNorm替代LayerNorm
    """

    def __init__(self, hidden_size, num_q_heads, num_kv_heads, mlp_ratio=4.0, dropout=0.1, drop_path=0.0):
        super().__init__()

        # V3: 使用PyTorch官方RMSNorm（2.8+）
        self.norm1 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.attn = GroupedQueryAttention(hidden_size, num_q_heads, num_kv_heads, dropout=dropout)

        self.norm2 = nn.RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),  # 添加MLP Dropout
            nn.Linear(mlp_hidden_dim, hidden_size),
            nn.Dropout(dropout)   # 添加输出 Dropout
        )

        # adaLN modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        # DropPath (Stochastic Depth)
        # 默认0.0时为Identity，兼容旧权重
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

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

        # Self-Attention with adaLN + DropPath
        x_norm1 = self.norm1(x)
        x_norm1 = x_norm1 * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_output = self.attn(x_norm1)
        x = x + self.drop_path(gate_msa.unsqueeze(1) * attn_output)

        # MLP with adaLN + DropPath
        x_norm2 = self.norm2(x)
        x_norm2 = x_norm2 * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_output = self.mlp(x_norm2)
        x = x + self.drop_path(gate_mlp.unsqueeze(1) * mlp_output)

        return x


class JaT_AudioSR_V3(nn.Module):
    """
    JaT-AudioSR V3 - Modern Architecture with RMSNorm

    V3 改进：
    - RMSNorm替代LayerNorm（更快、更稳定）
    - 保留GQA、RoPE、DropPath等V2特性
    """

    def __init__(self,
                 input_channels=1024,     # DAC latent channels
                 cond_channels=1024,      # LR condition latent channels
                 patch_len=4,             # Temporal patch length
                 hidden_size=1024,        # Transformer hidden size (LARGER!)
                 depth=16,                # Number of transformer blocks (DEEPER!)
                 num_q_heads=16,          # Query heads
                 num_kv_heads=4,          # KV heads (4x fewer for GQA)
                 bottleneck_dim=512,      # Bottleneck dimension (LARGER!)
                 mlp_ratio=4.0,
                 dropout=0.1,             # Dropout率（修复过拟合）
                 drop_path_rate=0.0):     # DropPath率（Stochastic Depth）
        super().__init__()

        self.input_channels = input_channels
        self.patch_len = patch_len
        self.hidden_size = hidden_size

        print("=" * 60)
        print("Initializing JaT-AudioSR V3 (Modern Architecture)")
        print("=" * 60)
        print(f"  DAC Latent shape: [B, {input_channels}, T/557]")
        print(f"  Patch length: {patch_len}")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Depth: {depth} blocks")
        print(f"  Query heads: {num_q_heads}, KV heads: {num_kv_heads} (GQA)")
        print(f"  Bottleneck dim: {bottleneck_dim}")
        print(f"  Dropout: {dropout}")
        print(f"  DropPath rate: {drop_path_rate}")

        # 1. Patch Embedding with bottleneck
        total_in_channels = input_channels + cond_channels
        self.patch_embed = BottleneckPatchEmbed1D(
            patch_len=patch_len,
            in_chans=total_in_channels,
            embed_dim=hidden_size,
            bottleneck_dim=bottleneck_dim
        )

        # 2. NO learnable positional embedding (RoPE handles it)
        # Just keep a dummy for sequence length check
        self.max_len = 2048

        # 3. Time Embedding
        self.t_embedder = nn.Sequential(
            TimeEmbedding(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # 4. Transformer Blocks with GQA
        # 使用线性递增的DropPath率：从0到drop_path_rate
        import torch
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            DiTBlock_GQA(hidden_size, num_q_heads, num_kv_heads, mlp_ratio,
                         dropout=dropout, drop_path=dpr[i])
            for i in range(depth)
        ])

        # 5. Final Layer (V3: 使用PyTorch官方RMSNorm)
        patch_out_dim = patch_len * input_channels
        self.final_layer = nn.Sequential(
            nn.RMSNorm(hidden_size, eps=1e-6),
            nn.Linear(hidden_size, patch_out_dim)
        )

        self.initialize_weights()

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"  Total params: {total_params / 1e6:.2f}M")
        print("=" * 60)

    def initialize_weights(self):
        """Initialize weights following best practices"""
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
        """
        P = self.patch_len
        N = x.shape[1]

        x = x.reshape(B, N, C, P)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, C, N * P)

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
            x_pred: [B, 1024, T] predicted clean latent
        """
        B, C, T_orig = x_t.shape

        # Pad to make T divisible by patch_len
        P = self.patch_len
        pad_len = (P - T_orig % P) % P
        if pad_len > 0:
            x_t = F.pad(x_t, (0, pad_len))
            x_cond = F.pad(x_cond, (0, pad_len))

        T = x_t.shape[-1]

        # 0. Concat conditioning
        x_in = torch.cat([x_t, x_cond], dim=1)  # [B, 2048, T]

        # 1. Patch Embedding
        x = self.patch_embed(x_in)  # [B, N, hidden_size]
        N = x.shape[1]

        # 2. Check sequence length (no positional embedding added)
        if N > self.max_len:
            raise ValueError(f"Sequence length {N} exceeds max_len {self.max_len}")

        # 3. Time Embedding
        t_emb = self.t_embedder(t)  # [B, hidden_size]

        # 4. Transformer Blocks (RoPE applied inside GQA)
        for block in self.blocks:
            x = block(x, t_emb)

        # 5. Final Projection
        x = self.final_layer(x)  # [B, N, C*P]

        # 6. Unpatchify
        x_pred = self.unpatchify(x, B, self.input_channels, T)

        # 7. Remove padding
        if pad_len > 0:
            x_pred = x_pred[:, :, :T_orig]

        return x_pred


def test_model():
    """Quick test of V3 model"""
    print("\n" + "=" * 60)
    print("Model V3 Test (with RMSNorm)")
    print("=" * 60)

    # Create model
    model = JaT_AudioSR_V3(
        input_channels=1024,
        cond_channels=1024,
        patch_len=4,
        hidden_size=1024,      # Larger
        depth=16,              # Deeper
        num_q_heads=16,
        num_kv_heads=4,        # GQA: 4x fewer KV heads
        bottleneck_dim=512,    # Larger bottleneck
        dropout=0.1            # Enable dropout
    )

    # Test forward pass
    B = 2
    T = 516
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
    print("✅ Model V2 test passed!")


if __name__ == '__main__':
    test_model()
