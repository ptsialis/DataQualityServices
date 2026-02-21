import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau



def compute_accuracy(logits, targets):
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()

def save_model(model, optimizer, epoch, acc, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch,
        'accuracy': acc
    }, path)

def load_model(model, optimizer, path, device='cpu'):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    return checkpoint['epoch'], checkpoint['accuracy']

def clip_gradients(model, max_norm):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

def warmup_unfreeze(model, epoch, warmup_epochs):
    if epoch == warmup_epochs:
        for param in model.encoder.parameters():
            param.requires_grad = True
        print(f"[Warmup] Unfroze encoder at epoch {epoch}.")

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_trainable(module, trainable):
    for param in module.parameters():
        param.requires_grad = trainable


def set_trainable(module, trainable):
    for param in module.parameters():
        param.requires_grad = trainable

def apply_unfreezing_schedule(model, epoch, args):
    if args.backbone != 'ResNet18':
        return

    k = 10 # Your chosen warmup (linear probe) duration

    # --- Step 1: Control the trainability of FEAT-specific layers (always trainable) ---
    set_trainable(model.prototype_attention_layers, True)
    set_trainable(model.auxiliary_transformer, True)
    if hasattr(model, 'prototype_norm_layers'):
        set_trainable(model.prototype_norm_layers, True)

    if epoch < k:  # Linear probe phase (epochs 0 to k-1)
        set_trainable(model.encoder, False) # Freeze the entire 'encoder' module.

        print(f"[Epoch {epoch}] Linear probe: Backbone (encoder and projection) frozen, only FEAT heads trainable.")

        current_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[Epoch {epoch} DEBUG LINEAR PROBE] Trainable parameters (expected low, only FEAT heads): {current_trainable_params}/{total_params} ({100.0 * current_trainable_params / total_params:.1f}%)")


    else: # Progressive unfreezing phase (epoch >= k)
        set_trainable(model.encoder, False) # Freeze the entire 'encoder' module again.

        current_trainable_params_after_freeze = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[Epoch {epoch} DEBUG BEFORE SELECTIVE UNFREEZE] Trainable parameters (expected low, only FEAT heads): {current_trainable_params_after_freeze}/{total_params} ({100.0 * current_trainable_params_after_freeze / total_params:.1f}%)")


        layers_to_unfreeze = []
        # Your unfreezing schedule remains the same
        if k <= epoch < k + 15: # Epochs 10, 25
            layers_to_unfreeze = ['7']
        elif k + 15 <= epoch < k + 25: # Epochs 25, 35
            layers_to_unfreeze = ['6', '7']
        elif k + 25 <= epoch < k + 35: # Epochs 35-45
            layers_to_unfreeze = ['5', '6', '7']
        elif epoch >= k + 35: # Epochs 17 onwards
            layers_to_unfreeze = ['all']

        if 'all' in layers_to_unfreeze:
            set_trainable(model.encoder, True)
            print(f"[Epoch {epoch}] Full backbone (encoder and projection) unfrozen.")
        else:
            for name, param in model.encoder.encoder.named_parameters():
                for layer in layers_to_unfreeze:
                    if name.startswith(layer + '.'):
                        param.requires_grad = True
                        print(f"Unfreezing: {name} (from layer {layer})")

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[Epoch {epoch}] Trainable parameters: {trainable_params}/{total_params} "
              f"({100.0 * trainable_params / total_params:.1f}%)")


    
def create_optimizer(model, args):
    if args.backbone == 'ResNet18':
        params = [
            {'params': model.encoder.encoder.parameters(), 'lr': args.lr * 0.1},
            {'params': model.encoder.projection.parameters(), 'lr': args.lr},
            {'params': model.prototype_attention_layers.parameters(), 'lr': args.lr},
            {'params': model.auxiliary_transformer.parameters(), 'lr': args.lr}
        ]
        return optim.SGD(
            params, 
            lr=args.lr, 
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            dampening=0,   # Required for Nesterov
            nesterov=True)
    else:
        if args.optimizer_type.lower() == 'sgd':
            return optim.SGD(
                model.parameters(),
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                nesterov=True)
        elif args.optimizer_type.lower() == 'adam':
            return optim.Adam(
                model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay)
        elif args.optimizer_type.lower() == 'adamw':
            return optim.AdamW(
                model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {args.optimizer_type}")


class EarlyStopper:
    def __init__(self, patience=100, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_acc = 0.0

    def check(self, val_acc):
        improved = False
        if val_acc > self.best_acc + self.min_delta:
            self.best_acc = val_acc
            self.counter = 0
            improved = True
        else:
            self.counter += 1
        return self.counter >= self.patience, improved
    
def save_config(args, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(f"{path}/config.txt", "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
            
def create_scheduler(optimizer, args):
    main_scheduler = ReduceLROnPlateau(optimizer, mode='max',factor=0.5, patience=10, verbose=True) # patience how many epochs to wait before reducing LR if no improvement in metric (e.g., val acc/loss
    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: min(1.0, (e + 1) / args.warmup_epochs))
    
    return main_scheduler, warmup_scheduler

