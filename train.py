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
import datetime
import math
import torch
import torch.nn.functional as F
import argparse
import time
import numpy as np
import tiktoken
import random
from models import ConfigurableTransformer, Config
from utils import HighwayVarianceMonitor, log_gradient_profile, DualLogger, init_csv_logger

def get_batch(data, cfg, device):
    ix = torch.randint(len(data) - cfg.seq_len - 1, (cfg.batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+cfg.seq_len].astype(np.int64)) for i in ix]).to(device)
    y = torch.stack([torch.from_numpy(data[i+1:i+cfg.seq_len+1].astype(np.int64)) for i in ix]).to(device)
    return x, y

def main():
    parser = argparse.ArgumentParser(description="UHT Training and Evaluation")
    parser.add_argument('--eval_only', action='store_true', help="Skip training and run validation on a checkpoint.")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to pre-trained weights.")
    parser.add_argument('--log_dir', type=str, default="./logs/", help="Directory to save logs and checkpoints.")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc = tiktoken.get_encoding("gpt2")
    Config.vocab_size = enc.n_vocab

    # --- EVALUATION ONLY MODE ---
    if args.eval_only:
        print("Running in Evaluation-Only Mode...")
        model = ConfigurableTransformer(Config).to(device)
        if args.checkpoint:
            print(f"Loading weights from {args.checkpoint}...")
            checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("[WARNING] No checkpoint provided. Evaluating untrained weights.")

        val_data = np.memmap("wiki_val.bin", dtype=np.uint16, mode='r')
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for _ in range(50): 
                X_val, Y_val = get_batch(val_data, Config, device)
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    # Uses final target scale of 1.0 for evaluation
                    val_logits = model(X_val, scale_factor=1.0)
                    batch_val_loss = F.cross_entropy(val_logits.view(-1, Config.vocab_size), Y_val.view(-1))
                val_loss += batch_val_loss.item()
            
            final_loss = val_loss / 50
            print(f"Validation Loss: {final_loss:.4f} | Perplexity: {math.exp(final_loss):.2f}")
        return # Exit after eval

    # --- FULL TRAINING LOOP ---
    tag = "UHT_Training_Run"
    log_path = f"{args.log_dir}{Config.num_layers}_LAYER_{tag}_RunLog_{timestamp}.txt"
    sys.stdout = DualLogger(log_path)
    sys.stderr = sys.stdout
    
    print(f"===========================================================")
    print(f"  {tag}: {device.type.upper()} | {Config.num_layers} LAYERS | LR {Config.lr}")
    print(f"===========================================================\n")

    train_data = np.memmap("wiki_train.bin", dtype=np.uint16, mode='r')
    val_data = np.memmap("wiki_val.bin", dtype=np.uint16, mode='r')

    steps_per_epoch = 10 #1000  
    seed_list = [42, 105]
    
    val_loss_history = {seed: [] for seed in seed_list}
    variance_history = {seed: [] for seed in seed_list} 

    csv_log_path = f"{args.log_dir}{Config.num_layers}_LAYER_{tag}_GradProfile_{timestamp}.csv"
    init_csv_logger(csv_log_path, total_layers=Config.num_layers)

    for seed in seed_list:
        print(f"\n{'='*60}\n STARTING RUN FOR SEED: {seed}\n{'='*60}")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        
        model = ConfigurableTransformer(Config).to(device)
        var_monitor = HighwayVarianceMonitor(model)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr, weight_decay=1e-4) 
        scaler = torch.amp.GradScaler('cuda')
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=Config.lr, epochs=Config.epochs, 
            steps_per_epoch=steps_per_epoch, pct_start=0.1, anneal_strategy='cos'
        )

        # Ensure the model always gets the full mathematical annealing timeline (e.g., 5000 steps)
        # regardless of how short we make the test epochs.
        base_steps_per_epoch = 1000
        adjusted_anneal_epochs = Config.anneal_epochs * (base_steps_per_epoch / steps_per_epoch)
        anneal_steps = int(adjusted_anneal_epochs * steps_per_epoch)

        best_val_loss_in_window = float('inf')
        checkpoint_path = f"{args.log_dir}{Config.num_layers}L_{tag}_Seed{seed}_Best.pt"

        for epoch in range(1, Config.epochs + 1):
            model.train()
            epoch_loss = 0.0
            epoch_start_time = time.time()
            
            for step in range(steps_per_epoch):
                X_batch, Y_batch = get_batch(train_data, Config, device)
                global_step = (epoch - 1) * steps_per_epoch + step
            
                # Dynamic Variance Annealing
                if global_step < anneal_steps:
                    progress = global_step / anneal_steps
                    current_L_multiplier = Config.dynamic_anneal_start - ((Config.dynamic_anneal_start - Config.final_anneal_mult) * progress) 
                    current_LC = Config.num_layers - ((Config.num_layers - 1) * progress)
                else:
                    current_L_multiplier = Config.final_anneal_mult
                    current_LC = 1
                
                current_scale_factor = math.sqrt(current_L_multiplier * current_LC)

                optimizer.zero_grad()
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(X_batch, scale_factor=current_scale_factor)
                    loss = F.cross_entropy(logits.view(-1, Config.vocab_size), Y_batch.view(-1))
                    
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                epoch_loss += loss.item()

                if step % 100 == 0:
                    log_gradient_profile(model, global_step, csv_log_path, total_layers=Config.num_layers, print_interval=12)
            
            # Validation Phase
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(50): 
                    X_val, Y_val = get_batch(val_data, Config, device)
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        val_logits = model(X_val, scale_factor=current_scale_factor)
                        batch_val_loss = F.cross_entropy(val_logits.view(-1, Config.vocab_size), Y_val.view(-1))
                    val_loss += batch_val_loss.item()
                val_loss /= 50
                
            last_layer_var = var_monitor.get_last_layer_variance()
            variance_history[seed].append(last_layer_var)
            
            print(f"Epoch {epoch:2d} | Train Loss: {epoch_loss/steps_per_epoch:.4f} | Val Loss: {val_loss:.4f} | Z_mem Var (L{Config.num_layers}): {last_layer_var:.2f}")

            if epoch > (Config.epochs - 20) and val_loss < best_val_loss_in_window:
                best_val_loss_in_window = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss
                }, checkpoint_path)
                print(f"   -> [CHECKPOINT] Saved new best weights (Val Loss: {val_loss:.4f})")

        var_monitor.remove_hooks()

if __name__ == "__main__":
    main()