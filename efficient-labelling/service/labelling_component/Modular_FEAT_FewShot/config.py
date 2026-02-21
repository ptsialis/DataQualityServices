# feat-pipeline/config.py
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Modular FEAT for Few-Shot Learning')

    # Dataset & Task
    parser.add_argument('--data_path', type=str, default=None, help='Root path to dataset (required in training mode)')
    parser.add_argument('--support_root', type=str, default=None, help='Path to folder containing support set (used in inference mode)')
    parser.add_argument('--query_root', type=str, default=None, help='Path to folder containing query set (used in inference mode)')
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--split_train', type=float, default=0.75)
    parser.add_argument('--split_val', type=float, default=0.15)
    parser.add_argument('--split_test', type=float, default=0.10)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'])
    parser.add_argument('--output_root', type=str, default= None, help='Output directory for classified images in inference')


    # Image & Augmentation
    parser.add_argument('--image_resize', type=int, default=94, help='Resize image before cropping')
    parser.add_argument('--image_crop', type=int, default=84, help='Final image size after cropping')
    parser.add_argument('--use_randaugment', action='store_true', help='Apply RandAugment in training')
    parser.add_argument('--use_color_jitter', action='store_true', help='Apply color jitter in training')
    parser.add_argument('--use_random_erasing', action='store_true', help='Apply random erasing in training')
    #added
    parser.add_argument('--use_random_affine', type=bool, default=False)
    parser.add_argument('--use_random_rotation', type=bool, default=False)
    parser.add_argument('--use_random_sharpness', type=bool, default=False)
    parser.add_argument('--use_random_vertical_flip', type=bool, default=False)
    parser.add_argument('--use_gaussian_blur', type=bool, default=False)

    # Backbone
    parser.add_argument('--backbone', type=str, default='ResNet18', choices=['ResNet18', 'ResNet12', 'ConvNet4'])
#     parser.add_argument('--freeze_encoder', action='store_true', help='Freeze encoder during initial training')
    parser.add_argument('--hidden_dim', type=int, default=640)
    parser.add_argument('--use_cosine', type=bool, default=True, help='Use cosine similarity instead of Euclidean')

    # FEAT Attention
    parser.add_argument('--proto_attn_layers', type=int, default=3)
    parser.add_argument('--proto_attn_heads', type=int, default=2)
    parser.add_argument('--aux_transformer_layers', type=int, default=2)
    parser.add_argument('--aux_transformer_heads', type=int, default=1)
    parser.add_argument('--aux_transformer_ffn_dim_factor', type=int, default=4)
    parser.add_argument('--dropout_rate', type=float, default=0.3) # used whereever there is training (in train_feat and inference while training the epochs)


    # Training
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--episodes_per_epoch', type=int, default=600, help='Training episodes per epoch') # for training
    parser.add_argument('--val_episodes_per_epoch', type=int, default=200, help='Validation episodes per epoch') # for eval in training
    parser.add_argument('--test_episodes_per_epoch', type=int, default=600, help='Test episodes for final eval') # if testing the model is required
    parser.add_argument('--num_episodes', type=int, default=150, help='Number of episodes for inference') # for inference
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--optimizer_type', type=str, default='SGD', choices=['SGD', 'Adam', 'AdamW'])
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--grad_clip', type=float, default=5.0, help='Max gradient norm for clipping')
    parser.add_argument('--norm_type', type=float, default=2.0, help='L2 norm')
    parser.add_argument('--min_delta', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--checkpoint_freq', type=int, default=5)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    
    
    # Temperatures and Loss
    parser.add_argument('--temperature1', type=float, default=1.0)
    parser.add_argument('--temperature2', type=float, default=1.0)
    parser.add_argument('--lambda_reg', type=float, default=0.2)

    # System
    parser.add_argument('--batch_size_episodes', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_path', type=str, default='checkpoints/feat')
    parser.add_argument('--load_checkpoint', type=str, help='Path to model checkpoint')

    # Dataset stats cache (optional reuse)
    parser.add_argument('--dataset_mean', type=float, nargs='+', default=None)
    parser.add_argument('--dataset_std', type=float, nargs='+', default=None)
    parser.add_argument('--use_support_stats', type=bool, default=True)
   

    # Inference special parameters
    parser.add_argument('--fine_tune_epochs', type=int, default=5, help='Epochs for fine-tuning (0 to disable)')
    parser.add_argument('--use_tta', action='store_true', help='Enable test-time augmentation')
    parser.add_argument('--normalization_stats', nargs=6, type=float, default=None, help='Normalization stats [mean_r, mean_g, mean_b, std_r, std_g, std_b]')
    parser.add_argument('--uncertain_thresh', type=float, default=0.25, help='Minimum confidence to accept prediction')
    

    return parser.parse_args()

# NOTE: args.device is set in train_feat.py:
# args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


