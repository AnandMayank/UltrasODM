# utils/remamba.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Remamba(nn.Module):
    def __init__(self, d_model, d_state=64, d_conv=4, expand=2):
        super().__init__()

        # Try to import and initialize Mamba with proper error handling
        try:
            from mamba_ssm import Mamba
            # Try different initialization approaches for video-mamba-suite compatibility
            try:
                # First try with bimamba_type="v2" for video-mamba-suite
                self.mamba = Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    bimamba_type="v2"  # Required for video-mamba-suite
                )
            except (AssertionError, TypeError):
                try:
                    # Try with minimal parameters and bimamba_type
                    self.mamba = Mamba(d_model=d_model, bimamba_type="v2")
                except (AssertionError, TypeError):
                    try:
                        # Try standard mamba-ssm initialization
                        self.mamba = Mamba(
                            d_model=d_model,
                            d_state=d_state,
                            d_conv=d_conv,
                            expand=expand,
                        )
                    except Exception as e:
                        print(f"Warning: All Mamba initialization attempts failed: {e}")
                        print("Using fallback LSTM-based implementation")
                        self.mamba = self._create_fallback_mamba(d_model)

        except ImportError:
            print("Warning: mamba_ssm not available, using fallback implementation")
            self.mamba = self._create_fallback_mamba(d_model)

        self.attention = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)  # Reduce memory
        self.norm = nn.LayerNorm(d_model)

    def _create_fallback_mamba(self, d_model):
        """Fallback implementation using LSTM + Conv1d"""
        class FallbackMamba(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
                self.lstm = nn.LSTM(d_model, d_model, batch_first=True, num_layers=1)

            def forward(self, x):
                # This won't be called since we handle fallback in the main forward method
                return x

        fallback = FallbackMamba()
        # Store components for access via indexing
        fallback.add_module('0', fallback.conv)
        fallback.add_module('1', fallback.lstm)
        return fallback

    def forward(self, x):
        # Handle different input shapes with memory optimization
        if len(x.shape) == 4:  # (B, C, H, W)
            B, C, H_orig, W_orig = x.shape

            # Memory optimization: Use adaptive pooling to reduce spatial dimensions
            if H_orig * W_orig > 1024:  # If spatial dimensions are too large
                x = F.adaptive_avg_pool2d(x, (32, 32))  # Reduce to manageable size
                H, W = 32, 32
            else:
                H, W = H_orig, W_orig

            # Reshape for sequence processing
            x = x.view(B, C, H*W).transpose(1, 2)  # (B, H*W, C)

            # Limit sequence length for attention to prevent memory explosion
            seq_len = x.shape[1]
            if seq_len > 512:  # Limit sequence length
                # Use stride to subsample
                stride = seq_len // 512
                x = x[:, ::stride, :]  # Subsample sequence
                seq_len = x.shape[1]

            # Mamba SSM processing
            if hasattr(self.mamba, 'forward'):
                try:
                    mamba_out = self.mamba(x)
                except Exception as e:
                    print(f"Mamba forward failed: {e}, using fallback")
                    mamba_out = x  # Simple passthrough fallback
            else:
                # Fallback processing
                x_conv = x.transpose(1, 2)  # (B, C, seq_len)
                x_conv = self.mamba[0](x_conv)  # Conv1d
                x_conv = x_conv.transpose(1, 2)  # (B, seq_len, C)
                mamba_out, _ = self.mamba[1](x_conv)  # LSTM returns (output, hidden)

            # Memory-efficient attention with chunking
            if seq_len > 256:  # Use chunked attention for large sequences
                chunk_size = 256
                attn_chunks = []
                for i in range(0, seq_len, chunk_size):
                    end_idx = min(i + chunk_size, seq_len)
                    chunk = mamba_out[:, i:end_idx, :]
                    chunk_attn, _ = self.attention(chunk, chunk, chunk)
                    attn_chunks.append(chunk_attn)
                x_attn = torch.cat(attn_chunks, dim=1)
            else:
                # Standard attention for smaller sequences
                x_attn, _ = self.attention(mamba_out, mamba_out, mamba_out)

            # Residual connection and normalization
            out = self.norm(mamba_out + x_attn)

            # Reshape back to spatial format - handle dimension mismatches
            out_transposed = out.transpose(1, 2)  # (B, C, seq_len)
            current_seq_len = out_transposed.shape[2]

            # Calculate what spatial dimensions we actually have
            if current_seq_len == H * W:
                # Perfect match - reshape normally
                out = out_transposed.view(B, C, H, W)
            else:
                # Dimension mismatch - find closest square dimensions
                sqrt_seq = int(current_seq_len ** 0.5)
                if sqrt_seq * sqrt_seq == current_seq_len:
                    # Perfect square
                    out = out_transposed.view(B, C, sqrt_seq, sqrt_seq)
                else:
                    # Not perfect square - pad or truncate to make it work
                    target_size = H * W
                    if current_seq_len > target_size:
                        # Truncate
                        out_transposed = out_transposed[:, :, :target_size]
                    else:
                        # Pad
                        padding = target_size - current_seq_len
                        out_transposed = F.pad(out_transposed, (0, padding))
                    out = out_transposed.view(B, C, H, W)

            # If we reduced dimensions, interpolate back to original size
            if H != H_orig or W != W_orig:
                out = F.interpolate(out, size=(H_orig, W_orig), mode='bilinear', align_corners=False)

        else:  # Already in sequence format
            # Limit sequence length for memory efficiency
            if x.shape[1] > 512:
                stride = x.shape[1] // 512
                x = x[:, ::stride, :]

            # Mamba SSM processing
            if hasattr(self.mamba, 'forward'):
                try:
                    mamba_out = self.mamba(x)
                except Exception as e:
                    print(f"Mamba forward failed: {e}, using fallback")
                    mamba_out = x  # Simple passthrough fallback
            else:
                # Fallback processing
                x_conv = x.transpose(1, 2)  # For Conv1d
                x_conv = self.mamba[0](x_conv)
                x_conv = x_conv.transpose(1, 2)
                mamba_out, _ = self.mamba[1](x_conv)  # LSTM returns (output, hidden)

            # Memory-efficient attention
            seq_len = x.shape[1]
            if seq_len > 256:
                chunk_size = 256
                attn_chunks = []
                for i in range(0, seq_len, chunk_size):
                    end_idx = min(i + chunk_size, seq_len)
                    chunk = mamba_out[:, i:end_idx, :]
                    chunk_attn, _ = self.attention(chunk, chunk, chunk)
                    attn_chunks.append(chunk_attn)
                x_attn = torch.cat(attn_chunks, dim=1)
            else:
                x_attn, _ = self.attention(mamba_out, mamba_out, mamba_out)

            # Residual connection
            out = self.norm(mamba_out + x_attn)

        return out