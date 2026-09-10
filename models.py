# ==============================================================================
# Unified Highway Transformer (UHT)
#
# Author: Anthony Luke Simon
# Year: 2026
# 
# Paper: Designing Better Transformers: Using Amorphous Network Proxies to 
#        Engineer the Unified Highway Transformer
# DOI: 10.6084/m9.figshare.33510313
#
# Notice: Aspects of the Unified Highway Transformer architecture and test 
# methodology are the subject of pending patent applications filed by the author. 
# This code is provided for academic review and reproducibility and for use under
# the license on the site in which it is posted (e.g., from GitHub, MIT License).
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Config:
    seq_len = 512          
    
    # --- The Bottleneck ---
    embed_dim = 64          
    num_heads = 4           
    ffn_dim = 256           
    num_layers = 120        
    
    # --- Training Hyperparameters ---
    lr = 0.001 
    epochs = 50             
    batch_size = 16         # Change to 8 in train.py for T4 gradient accumulation
    dropout = 0.05
    
    # --- Structural Memory Dials ---
    use_single_tap_highway = True 
    use_z_mem_highway = True  
    strided_projections = True 
    forward_clamp_val = None      
    
    # --- Dynamic Variance Annealing ---
    dynamic_anneal_start = 8.0
    final_anneal_mult = 1.0 
    anneal_epochs = 5 

class ConfigurableTransformerBlock(nn.Module):
    def __init__(self, cfg, layer_idx=0): 
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx 
        
        self.ln1 = nn.LayerNorm(cfg.embed_dim)
        self.attn = nn.MultiheadAttention(cfg.embed_dim, cfg.num_heads, dropout=cfg.dropout, batch_first=True)
        
        self.ln2 = nn.LayerNorm(cfg.embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.ffn_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ffn_dim, cfg.embed_dim),
            nn.Dropout(cfg.dropout)
        )
        
        if cfg.use_single_tap_highway:
            self.univ_layer_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)
            
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, z_mem=None, scale_factor=None):
        B, T, C = x.size()
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
    
        if scale_factor is None:
            scale_factor = math.sqrt(2 * self.cfg.num_layers)
        
        # --- 1. ATTENTION READ ---
        norm_x = self.ln1(x)
        attn_input = (norm_x + z_mem) if z_mem is not None else norm_x
        attn_out, _ = self.attn(attn_input, attn_input, attn_input, is_causal=True, attn_mask=causal_mask)
        x = x + self.dropout(attn_out)
    
        # --- 2. FFN READ ---
        norm_x2 = self.ln2(x)
        ffn_input = (norm_x2 + z_mem) if z_mem is not None else norm_x2
        ffn_out = self.ffn(ffn_input)
        x = x + self.dropout(ffn_out)
        
        # === UNIFIED POST-LAYER FOLD ===
        if z_mem is not None and self.cfg.use_single_tap_highway:
            should_update_memory = True
            if getattr(self.cfg, 'strided_projections', False):
                if self.layer_idx % 2 == 0:
                    should_update_memory = False 
            
            if should_update_memory:
                layer_injection = F.leaky_relu(self.univ_layer_proj(x), negative_slope=0.1)
                z_mem = z_mem + (layer_injection / scale_factor)
        
        return x, z_mem

class ConfigurableTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_embed = nn.Embedding(cfg.seq_len, cfg.embed_dim)
        
        self.layers = nn.ModuleList([
            ConfigurableTransformerBlock(cfg, layer_idx=i) 
            for i in range(cfg.num_layers)                    
        ])
        
        self.ln_f = nn.LayerNorm(cfg.embed_dim)
        self.vocab_proj = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)
        self.vocab_proj.weight = self.token_embed.weight
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, scale_factor=None):
        B, T = x.size()
        positions = torch.arange(0, T, device=x.device).unsqueeze(0).expand(B, T)
    
        x = self.dropout(self.token_embed(x) + self.pos_embed(positions))
        z_mem = x.clone() if getattr(self.cfg, 'use_z_mem_highway', True) else None
            
        # Clean, native forward pass
        for layer in self.layers:
            x, z_mem = layer(x, z_mem, scale_factor=scale_factor)
        
        x = self.ln_f(x)
        return self.vocab_proj(x)