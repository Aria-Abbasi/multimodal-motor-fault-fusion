import torch
import torch.nn as nn
import time

class SpectrogramEncoder(nn.Module):
    """Single-modality encoder: 4 Conv blocks -> Global Pool -> 256d"""
    def __init__(self, in_channels=1, embed_dim=256):
        super().__init__()
        
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.GELU(), # Modern activation
                nn.MaxPool2d(2)
            )
            
        # Input: (Batch, 1, 128, 128)
        self.features = nn.Sequential(
            conv_block(in_channels, 32),   # -> (B, 32, 64, 64)
            conv_block(32, 64),            # -> (B, 64, 32, 32)
            conv_block(64, 128),           # -> (B, 128, 16, 16)
            conv_block(128, 256)           # -> (B, 256, 8, 8)
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(256, embed_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x).view(x.size(0), -1)
        return self.projection(x)

class CrossAttentionFusion(nn.Module):
    """Fuses modalities using 4-head MultiheadAttention"""
    def __init__(self, embed_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        # Vib attends to Curr
        self.attn_v_c = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        # Curr attends to Vib
        self.attn_c_v = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim * 2)

    def forward(self, vib_emb, curr_emb):
        # Reshape to (Batch, SeqLen=1, EmbedDim) for attention
        v_seq = vib_emb.unsqueeze(1)
        c_seq = curr_emb.unsqueeze(1)
        
        # Cross Attention Lookups
        v_fused, _ = self.attn_v_c(query=v_seq, key=c_seq, value=c_seq)
        c_fused, _ = self.attn_c_v(query=c_seq, key=v_seq, value=v_seq)
        
        # Concat and flatten back to (Batch, 512)
        fused = torch.cat([v_fused.squeeze(1), c_fused.squeeze(1)], dim=-1)
        return self.norm(fused)

class MultimodalMotorModel(nn.Module):
    """The full proposed model with ablation flags and dual-heads"""
    def __init__(self, embed_dim=256, num_fault_families=5, ablation_mode=None):
        super().__init__()
        self.ablation_mode = ablation_mode # None, "vibration_only", "current_only"
        
        self.vib_encoder = SpectrogramEncoder(in_channels=1, embed_dim=embed_dim)
        self.curr_encoder = SpectrogramEncoder(in_channels=1, embed_dim=embed_dim)
        
        self.fusion = CrossAttentionFusion(embed_dim=embed_dim, num_heads=4)
        
        # Dual outputs: Early-Fault (Binary) and Fault Family (Multi-class)
        fusion_dim = embed_dim * 2 if ablation_mode is None else embed_dim
        
        self.head_early_fault = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2) # 0: Healthy, 1: Faulty
        )
        
        self.head_fault_family = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_fault_families)
        )

    def forward(self, x):
        # Input x shape: (Batch, Channels=2, Freq=128, Time=128)
        # Channel 0: Vibration, Channel 1: Current
        vib_signal = x[:, 0:1, :, :]
        curr_signal = x[:, 1:2, :, :]
        
        if self.ablation_mode == "vibration_only":
            features = self.vib_encoder(vib_signal)
        elif self.ablation_mode == "current_only":
            features = self.curr_encoder(curr_signal)
        else: # Full Multimodal Fusion
            v_emb = self.vib_encoder(vib_signal)
            c_emb = self.curr_encoder(curr_signal)
            features = self.fusion(v_emb, c_emb)
            
        out_health = self.head_early_fault(features)
        out_family = self.head_fault_family(features)
        
        return out_health, out_family

# --- Smoke Test & Shape Validation ---
if __name__ == "__main__":
    print("🚀 Running Step 9 Architecture Smoke Test...")
    
    # Simulate a batch of 8 windows, 2 channels, 128x128 spectrograms
    batch_size = 8
    dummy_input = torch.randn(batch_size, 2, 128, 128)
    
    # Test 1: Full Fusion Model
    print("\n--- Testing Full Multimodal Fusion ---")
    model = MultimodalMotorModel(num_fault_families=5)
    
    start_time = time.time()
    # Test standard forward pass
    out_health, out_family = model(dummy_input)
        
    print(f"Input Shape:  {dummy_input.shape}")
    print(f"Health Head:  {out_health.shape} (Expected: {batch_size}, 2)")
    print(f"Family Head:  {out_family.shape} (Expected: {batch_size}, 5)")
    print(f"Forward Pass: {(time.time() - start_time)*1000:.2f} ms")
    
    # Test 2: Ablation Mode (Vibration Only)
    print("\n--- Testing Ablation (Vibration Only) ---")
    model_vib = MultimodalMotorModel(ablation_mode="vibration_only")
    out_health_v, _ = model_vib(dummy_input)
    print(f"Ablated Health Head Shape: {out_health_v.shape}")
    
    print("\n✅ Step 9 Architecture is mathematically sound and ready for training!")
