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

import os
import sys
import csv
import torch

# =====================================================================
# 1. Terminal / File Logger
# =====================================================================
class DualLogger(object):
    """
    Duplicates stdout and stderr to both the active console and a disk log file.
    Flushes immediately to ensure logs survive unexpected crashes.
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# =====================================================================
# 2. Deep Highway Variance Diagnostic Tool
# =====================================================================
class HighwayVarianceMonitor:
    """
    Attaches forward hooks to projection modules across all layers
    to monitor variance growth along the un-normalized memory bus (Z_mem).
    """
    def __init__(self, model):
        self.variances = {}
        self.hooks = []
        
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'univ_layer_proj'):
                target = layer.univ_layer_proj
                hook = target.register_forward_hook(self._variance_hook(i))
                self.hooks.append(hook)

    def _variance_hook(self, layer_id):
        def hook_fn(module, input, output):
            # Computed detached in float32 for numerical stability
            current_var = output.detach().float().var().item()
            self.variances[layer_id] = current_var
        return hook_fn

    def get_logs(self):
        return self.variances

    def get_last_layer_variance(self):
        if not self.variances:
            return 0.0
        last_id = max(self.variances.keys())
        return self.variances[last_id]

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


# =====================================================================
# 3. Layer-by-Layer Gradient Diagnostics
# =====================================================================
def get_layer_grad_norm(layer):
    """Calculates the combined L2 gradient norm across all parameters in a block."""
    total_norm = 0.0
    for p in layer.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def init_csv_logger(filepath, total_layers=120):
    """Initializes CSV header structure for layer-depth gradient logging."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None
    headers = ['Step'] + [f'L{i}_Grad' for i in range(total_layers)]
    with open(filepath, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
    print(f"[SYSTEM] CSV Logger Initialized: {filepath}")


def log_gradient_profile(model, step, filepath, total_layers=120, print_interval=12):
    """Logs full layer-by-layer gradient norms to disk and prints selected deciles."""
    norms = [get_layer_grad_norm(model.layers[i]) for i in range(total_layers)]
    
    # 1. Disk persistence (open, append, flush/close)
    with open(filepath, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([step] + [f"{n:.4f}" for n in norms])
        
    # 2. Live console decile readout
    profile_str = " | ".join([f"L{i}: {norms[i]:.4f}" for i in range(0, total_layers, print_interval)])
    print(f"   Step {step:4d} | Decile Profile -> {profile_str}")