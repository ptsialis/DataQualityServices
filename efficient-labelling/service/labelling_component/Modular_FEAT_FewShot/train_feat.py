import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import random
from tqdm import tqdm
from config import get_args
from models.feat import FEAT
from models.backbones import get_backbone
from data.loader import get_fsl_loader
from utils.training_utils import EarlyStopper, save_config, apply_unfreezing_schedule, create_optimizer, create_scheduler
#from newProject.data.metaAlbum14 import get_metaalbumbirds_loader # To ensure that the code works well on the original loader!


def run_epoch(model, data_loader, optimizer, device, epoch, args, mode='train'):
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    if mode == 'train':
        episodes = args.episodes_per_epoch
    else:
        episodes = args.val_episodes_per_epoch  

    is_train = mode == 'train'
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch} [{mode.capitalize()}]", total=min(len(data_loader), episodes))
    
    for i, batch in enumerate(progress_bar):
        if i >= episodes:
            break
            
        support_x = batch['train']['inputs'].to(device)
        support_y = batch['train']['targets'].to(device)
        query_x = batch['test']['inputs'].to(device)
        query_y = batch['test']['targets'].to(device)
        
        
        if is_train:
            optimizer.zero_grad()
            logits, logits_reg = model(support_x, support_y, query_x, query_y, mode='train')
            if i == 0:
                print(f'Logging logits mean {logits.mean()} and logits std {logits.std()}')
            # Main loss with smoothing
            loss_main = F.cross_entropy(logits, query_y, label_smoothing=args.label_smoothing)
            
            # Auxiliary loss with smoothing
            aux_labels = torch.repeat_interleave(
                torch.arange(args.way, device=device),
                args.shot + args.query
            )
            
            assert logits_reg.shape[0] == aux_labels.shape[0], f"Mismatch in aux logits ({logits_reg.shape[0]}) and aux labels ({aux_labels.shape[0]}) count"
            assert logits_reg.shape[0] == args.way * (args.shot + args.query), (f"Expected aux logits shape ({args.way} x ({args.shot}+{args.query})) = "f"{args.way * (args.shot + args.query)}, but got {logits_reg.shape[0]}")
            loss_aux = F.cross_entropy(logits_reg, aux_labels, label_smoothing=args.label_smoothing)
            
            loss = loss_main + args.lambda_reg * loss_aux
            loss.backward()
            
            # Gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, args.norm_type)
            
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(support_x, support_y, query_x, query_y, mode='eval')
                loss = F.cross_entropy(logits, query_y, label_smoothing=args.label_smoothing)
        
        # Calculate accuracy (match original)
        preds = logits.argmax(dim=1)
        correct = (preds == query_y).sum().item()
        total = query_y.size(0)
        acc = correct / total
        
        total_loss += loss.item()
        total_correct += correct
        total_samples += total

        progress_bar.set_postfix(loss=loss.item(), acc=acc) #shows progress per batch
        #progress_bar.set_postfix(loss=total_loss / total_samples,acc=total_correct / total_samples) # shows progress over the whole epoch (not correct!)

    # Return global epoch metrics (matches original)
    avg_loss = total_loss / (i + 1)
    epoch_acc = total_correct / total_samples
    print(f'Epoch{epoch} loss {avg_loss}, accuracy {epoch_acc}')
    return avg_loss, epoch_acc    


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode == 'train' and args.data_path is None:
        raise ValueError("`--data_path` is required in training mode.")
    
    # Setup reproducibility
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Create save directory
    os.makedirs(args.save_path, exist_ok=True)
    save_config(args, args.save_path)
    writer = SummaryWriter(log_dir=os.path.join(args.save_path, 'logs'))
    

    encoder = get_backbone(args.backbone, args.hidden_dim, args.dropout_rate).to(device)
    model = FEAT(
        encoder=encoder,
        hidden_dim=args.hidden_dim,
        temp1=args.temperature1,
        temp2=args.temperature2,
        use_cosine=args.use_cosine,
        dropout_rate=args.dropout_rate,
        proto_attn_layers=args.proto_attn_layers,
        proto_attn_heads=args.proto_attn_heads,
        aux_transformer_layers=args.aux_transformer_layers,
        aux_transformer_heads=args.aux_transformer_heads,
        aux_transformer_ffn_dim_factor=args.aux_transformer_ffn_dim_factor
    ).to(device)

    optimizer = create_optimizer(model, args)
    main_scheduler, warmup_scheduler = create_scheduler(optimizer, args)

    train_loader = get_fsl_loader(args,'train')
    val_loader = get_fsl_loader(args,'val')


    best_val_acc = 0.0
    early_stopper = EarlyStopper(patience=args.patience, min_delta=args.min_delta)
    
    for epoch in range(1, args.epochs + 1):
        # Apply unfreezing schedule if needed
        if args.backbone == 'ResNet18':
            apply_unfreezing_schedule(model, epoch, args)
        
        # Training phase
        model.train()
        train_loss, train_acc = run_epoch(
            model, train_loader, optimizer, device, epoch, args, mode='train'
        )
        
        # Validation phase
        model.eval()
        val_loss, val_acc = run_epoch(
            model, val_loader, None, device, epoch, args, mode='eval'
        )
        
        # Update schedulers
        if epoch < args.warmup_epochs:
            warmup_scheduler.step()
        else:
            main_scheduler.step(val_acc)  # Step on validation accuracy
        
        # Logging
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/Acc', train_acc, epoch)
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Val/Acc', val_acc, epoch)
        
        # Check early stopping and save best model
        stop, improved = early_stopper.check(val_acc)
        if improved:
            print(f'imporved in epoch {epoch} and best value is {val_acc}')
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.save_path, "best_model.pth"))
        
        if stop:
            print(f"Early stopping at epoch {epoch}, best acc: {best_val_acc:.2f}%")
            break
            
        # Periodic checkpointing
        if epoch % args.checkpoint_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_acc': val_acc
            }, os.path.join(args.save_path, f"checkpoint_epoch_{epoch}.pth"))
        
       # Final save and cleanup
    torch.save(model.state_dict(), os.path.join(args.save_path, "final_model.pth"))
    writer.close()
    print(f"Training complete. Best validation accuracy: {best_val_acc*100:.2f}%") 

if __name__ == '__main__':
    main()
