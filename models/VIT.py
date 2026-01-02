import torch
import torch.nn as nn
from torch import Tensor

try:
    from transformers import ViTModel, ViTConfig
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: transformers library not available. Install with: pip install transformers")


class PatchEmbedding(nn.Module):
    """
    Splits an image into patches and embeds them.
    """

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        """
        Args:
            img_size: Input image size (assumes square images)
            patch_size: Size of each patch
            in_channels: Number of input channels (3 for RGB)
            embed_dim: Embedding dimension
        """
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Convolutional layer to create patch embeddings
        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, N, D] where N = number of patches, D = embed_dim
        """
        x = self.projection(x)  # [B, D, H/P, W/P]
        x = x.flatten(2)  # [B, D, N]
        x = x.transpose(1, 2)  # [B, N, D]
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.
    """

    def __init__(self, embed_dim=768, num_heads=12, dropout=0.0):
        """
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attention_dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.projection_dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [B, N, D]
        Returns:
            [B, N, D]
        """
        batch_size, n_patches, embed_dim = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x)  # [B, N, 3*D]
        qkv = qkv.reshape(batch_size, n_patches, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D/H]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)  # [B, H, N, N]
        attention = torch.softmax(scores, dim=-1)
        attention = self.attention_dropout(attention)

        # Apply attention to values
        out = attention @ v  # [B, H, N, D/H]
        out = out.transpose(1, 2)  # [B, N, H, D/H]
        out = out.reshape(batch_size, n_patches, embed_dim)  # [B, N, D]

        # Final projection
        out = self.projection(out)
        out = self.projection_dropout(out)

        return out


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (Feed-Forward Network) used in Transformer blocks.
    """

    def __init__(self, embed_dim=768, mlp_ratio=4.0, dropout=0.0):
        """
        Args:
            embed_dim: Embedding dimension
            mlp_ratio: Ratio of hidden dimension to embedding dimension
            dropout: Dropout rate
        """
        super(MLP, self).__init__()
        hidden_dim = int(embed_dim * mlp_ratio)

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder Block: Self-Attention + MLP with residual connections.
    """

    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.0):
        """
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dimension to embedding dimension
            dropout: Dropout rate
        """
        super(TransformerEncoderBlock, self).__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        # Self-attention with residual connection
        x = x + self.attention(self.norm1(x))

        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))

        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) for image classification.
    Based on "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
        use_class_token=True,
    ):
        """
        Args:
            img_size: Input image size (assumes square images)
            patch_size: Size of each patch
            in_channels: Number of input channels (3 for RGB)
            num_classes: Number of output classes
            embed_dim: Embedding dimension
            depth: Number of transformer encoder blocks
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dimension to embedding dimension
            dropout: Dropout rate
            use_class_token: Whether to use a learnable class token
        """
        super(VisionTransformer, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.use_class_token = use_class_token

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches

        # Class token (learnable)
        if use_class_token:
            self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            n_tokens = n_patches + 1
        else:
            n_tokens = n_patches

        # Positional embeddings (learnable)
        self.positional_embedding = nn.Parameter(torch.zeros(1, n_tokens, embed_dim))
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder blocks
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )

        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using truncated normal distribution."""
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)
        if self.use_class_token:
            nn.init.trunc_normal_(self.class_token, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, num_classes]
        """
        batch_size = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # [B, N, D]

        # Add class token
        if self.use_class_token:
            class_tokens = self.class_token.expand(batch_size, -1, -1)  # [B, 1, D]
            x = torch.cat([class_tokens, x], dim=1)  # [B, N+1, D]

        # Add positional embedding
        x = x + self.positional_embedding
        x = self.dropout(x)

        # Transformer encoder blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Classification: use class token if available, otherwise global average pooling
        if self.use_class_token:
            x = x[:, 0]  # [B, D]
        else:
            x = x.mean(dim=1)  # [B, D]

        x = self.head(x)  # [B, num_classes]

        return x


class VisionTransformerPretrained(nn.Module):
    """
    Vision Transformer using Hugging Face pre-trained weights.
    Supports models like 'google/vit-base-patch16-224-in21k' for fine-tuning.
    """

    def __init__(
        self,
        model_name="google/vit-base-patch16-224-in21k",
        num_classes=1000,
        freeze_backbone=False,
        dropout=0.1,
    ):
        """
        Args:
            model_name: Hugging Face model identifier
            num_classes: Number of output classes for your task
            freeze_backbone: If True, freeze the backbone and only train the classifier
            dropout: Dropout rate for the classifier head
        """
        super(VisionTransformerPretrained, self).__init__()

        if not HF_AVAILABLE:
            raise ImportError(
                "transformers library is required for pretrained models. "
                "Install with: pip install transformers"
            )

        # Load pre-trained ViT model (without classification head)
        self.vit = ViTModel.from_pretrained(model_name, add_pooler_layer=False)

        # Get the hidden size from the config
        self.hidden_size = self.vit.config.hidden_size

        # Optionally freeze the backbone
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
            print(f"Frozen backbone weights. Only training classifier head.")

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, num_classes),
        )

        # Initialize classifier weights
        self._init_classifier()

    def _init_classifier(self):
        """Initialize the classifier head."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor):
        """
        Args:
            x: [B, C, H, W] - Input images
        Returns:
            [B, num_classes] - Class logits
        """
        # Pass through ViT backbone
        outputs = self.vit(pixel_values=x)

        # Get the [CLS] token representation (first token)
        cls_token = outputs.last_hidden_state[:, 0]

        # Pass through classifier head
        logits = self.classifier(cls_token)

        return logits

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters for full fine-tuning."""
        for param in self.vit.parameters():
            param.requires_grad = True
        print("Unfrozen backbone. Training all parameters.")

    def freeze_backbone(self):
        """Freeze backbone parameters (only train classifier)."""
        for param in self.vit.parameters():
            param.requires_grad = False
        print("Frozen backbone. Only training classifier head.")


# Common ViT configurations
def ViT_Tiny(num_classes=1000, img_size=224):
    """ViT-Tiny: embed_dim=192, depth=12, heads=3"""
    return VisionTransformer(
        img_size=img_size,
        patch_size=16,
        num_classes=num_classes,
        embed_dim=192,
        depth=12,
        num_heads=3,
    )


def ViT_Small(num_classes=1000, img_size=224):
    """ViT-Small: embed_dim=384, depth=12, heads=6"""
    return VisionTransformer(
        img_size=img_size,
        patch_size=16,
        num_classes=num_classes,
        embed_dim=384,
        depth=12,
        num_heads=6,
    )


def ViT_Base(num_classes=1000, img_size=224):
    """ViT-Base: embed_dim=768, depth=12, heads=12"""
    return VisionTransformer(
        img_size=img_size,
        patch_size=16,
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
    )


def ViT_Large(num_classes=1000, img_size=224):
    """ViT-Large: embed_dim=1024, depth=24, heads=16"""
    return VisionTransformer(
        img_size=img_size,
        patch_size=16,
        num_classes=num_classes,
        embed_dim=1024,
        depth=24,
        num_heads=16,
    )


def ViT_Huge(num_classes=1000, img_size=224):
    """ViT-Huge: embed_dim=1280, depth=32, heads=16"""
    return VisionTransformer(
        img_size=img_size,
        patch_size=14,
        num_classes=num_classes,
        embed_dim=1280,
        depth=32,
        num_heads=16,
    )
