"""
wsa_train.py

Main training script for WSA (Wang-Sheeley-Arge) map prediction.

This script orchestrates the complete training pipeline:
  - Loads configuration from wsa_config.yaml
  - Creates train/val/test datasets with specified CR lists
  - Loads pre-trained Surya encoder and attaches WSA decoder head
  - Sets up PyTorch Lightning training with logging and checkpointing
  - Trains the model on WSA map prediction task

Usage:
    python wsa_train.py
    
    Optional CLI arguments:
    python wsa_train.py --config ./configs/wsa_config.yaml --cuda-device 0
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

import torch
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader

# Add paths for imports
sys.path.append("../../")
sys.path.append("../../Surya")

from surya.utils.data import build_scalers
from wsa_surya_training.datasets.wsa_dataset import WSAImageDataset
from wsa_surya_training.metrics.wsa_metrics import WSAMetrics
from wsa_surya_training.models.wsa_model_head import WSAModel
from wsa_surya_training.lightning_modules.wsa_lightning_module import WSALightningModule


# ============================================================================
# Configuration & Setup
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train WSA map prediction model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/home/momchilmolnar/workspace/surya_workshop_personal/surya_workshop/downstream_apps/template/wsa_surya_training/configs/wsa_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=1,
        help="CUDA device ID to use"
    )
    parser.add_argument(
        "--unfreeze-encoder",
        action="store_true",
        help="Unfreeze encoder for fine-tuning (default: frozen)"
    )
    return parser.parse_args()


def setup_environment(cuda_device: int):
    """Configure environment variables for training."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    
    # Configure W&B directories
    os.environ["WANDB_DIR"] = "./wandb/wandb_logs"
    os.environ["WANDB_CACHE_DIR"] = "./wandb/wandb_cache"
    os.environ["WANDB_CONFIG_DIR"] = "./wandb/wandb_config"
    os.environ["TMPDIR"] = "./wandb/wandb_tmp"
    
    # Create directories if they don't exist
    for dir_path in [
        os.environ["WANDB_DIR"],
        os.environ["WANDB_CACHE_DIR"],
        os.environ["WANDB_CONFIG_DIR"],
        os.environ["TMPDIR"],
    ]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Set precision and reproducibility
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(42, workers=True)
    
    print(f"✅ Environment configured (CUDA device: {cuda_device})")


def load_config(config_path: str) -> dict:
    """Load and parse YAML configuration file."""
    print(f"📋 Loading configuration from {config_path}...")
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Load scalers configuration
        scalers_path = config["data"]["scalers_path"]
        with open(scalers_path, "r") as f:
            config["data"]["scalers"] = yaml.safe_load(f)
        
        print("✅ Configuration loaded successfully!")
        return config
    
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"Make sure config file exists at {config_path}")
        raise


# ============================================================================
# Dataset Creation
# ============================================================================

def create_datasets(config: dict, scalers: dict) -> tuple:
    """Create train, val, and test datasets from configuration."""
    print("\n📊 Creating datasets...")
    
    # Common parameters
    dataset_kwargs = {
        "surya_index_path": config["data"]["surya_index_path"],
        "wsa_map_dir": config["data"]["wsa_map_dir"],
        "scalers": scalers,
        "channels": config["data"]["channels"],
        "normalize_wsa": config.get("normalize_wsa_maps", True),
        "tolerance_days": config["data"]["cr_tolerance_days"],
        "s3_use_simplecache": config["data"]["s3_use_simplecache"],
        "s3_cache_dir": config["data"]["s3_cache_dir"],
    }
    
    # Create train dataset
    train_dataset = WSAImageDataset(
        cr_list=config["data"]["train_crs"],
        phase="train",
        **dataset_kwargs
    )
    
    # Create validation dataset
    val_dataset = WSAImageDataset(
        cr_list=config["data"]["val_crs"],
        phase="val",
        **dataset_kwargs
    )
    
    # Create test dataset
    test_dataset = WSAImageDataset(
        cr_list=config["data"]["test_crs"],
        phase="test",
        **dataset_kwargs
    )
    
    print(f"✅ Datasets created:")
    print(f"   - Train: {len(train_dataset)} samples")
    print(f"   - Val:   {len(val_dataset)} samples")
    print(f"   - Test:  {len(test_dataset)} samples")
    
    return train_dataset, val_dataset, test_dataset


def create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size: int,
    num_workers: int = 4
) -> tuple:
    """Create PyTorch DataLoaders from datasets."""
    print("\n🔄 Creating dataloaders...")
    
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "multiprocessing_context": "spawn",
        "persistent_workers": True,
        "pin_memory": True,
    }
    
    train_loader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        **loader_kwargs
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        shuffle=False,
        **loader_kwargs
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        shuffle=False,
        **loader_kwargs
    )
    
    print(f"✅ Dataloaders created (batch_size: {batch_size})")
    
    return train_loader, val_loader, test_loader


# ============================================================================
# Model Setup
# ============================================================================

def load_surya_encoder(checkpoint_path: str, device: str = "cuda"):
    """Load pre-trained Surya encoder from checkpoint."""
    print(f"\n🧠 Loading Surya encoder from {checkpoint_path}...")
    
    try:
        # Import Surya model (adjust based on actual Surya API)
        from surya.models.helio_spectformer import HelioBERT
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Initialize encoder
        encoder = HelioBERT(
            img_size=4096,
            patch_size=16,
            in_channels=1,  # Single AIA 193 channel
            embed_dim=1280,
            depth=10,
            num_heads=16,
            mlp_ratio=4.0,
            window_size=2,
        )
        
        # Load weights (adjust key mapping if needed)
        if "model" in checkpoint:
            encoder.load_state_dict(checkpoint["model"], strict=False)
        else:
            encoder.load_state_dict(checkpoint, strict=False)
        
        encoder = encoder.to(device)
        print("✅ Surya encoder loaded successfully!")
        return encoder
    
    except Exception as e:
        print(f"❌ Error loading encoder: {e}")
        print("Falling back to randomly initialized encoder...")
        
        # Fallback: initialize randomly (for testing)
        from surya.models.helio_spectformer import HelioSpectFormer
        encoder = HelioSpectFormer(
            img_size=4096,
            patch_size=16,
            in_chans=1,
            embed_dim=1280,
            depth=10,
            num_heads=16,
            mlp_ratio=4.0,
            dp_rank=1, 
            time_embedding={'type': 'linear', 'time_dim':1},
            drop_rate = 0.1, 
            window_size=1, 
            n_spectral_blocks=1
        ).to(device)
        return encoder


def create_wsa_model(
    encoder,
    encoder_out_channels: int = 1280,
    freeze_encoder: bool = True,
) -> WSAModel:
    """Create WSA model with encoder and decoder head."""
    print("\n🏗️  Creating WSA model...")
    
    model = WSAModel(
        encoder=encoder,
        encoder_out_channels=encoder_out_channels,
        freeze_encoder=freeze_encoder,
    )
    
    freeze_status = "frozen" if freeze_encoder else "trainable"
    print(f"✅ WSA model created (encoder: {freeze_status})")
    
    return model


# ============================================================================
# Metrics and Logging
# ============================================================================

def create_metrics():
    """Create metric calculators."""
    print("\n📊 Creating metrics...")
    
    train_loss_metrics = WSAMetrics("train_loss")
    train_eval_metrics = WSAMetrics("train_metrics")
    val_eval_metrics = WSAMetrics("val_metrics")
    
    print("✅ Metrics created (MAE for train_loss, train_metrics, val_metrics)")
    
    return train_loss_metrics, train_eval_metrics, val_eval_metrics


def setup_loggers(config: dict) -> tuple:
    """Setup W&B and CSV loggers."""
    print("\n📝 Setting up loggers...")
    
    # Generate run name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config['wandb']['run_name']}_{timestamp}"
    
    # W&B Logger
    wandb_logger = WandbLogger(
        entity=config["wandb"]["entity"],
        project=config["wandb"]["project"],
        name=run_name,
        log_model=False,
        save_dir=os.environ.get("WANDB_DIR", "./wandb/wandb_tmp"),
    )
    
    # CSV Logger
    csv_logger = CSVLogger("runs", name="wsa_training")
    
    print(f"✅ Loggers created:")
    print(f"   - W&B: {config['wandb']['project']} / {run_name}")
    print(f"   - CSV: runs/wsa_training/")
    
    return wandb_logger, csv_logger


# ============================================================================
# Training
# ============================================================================

def train(
    config: dict,
    train_loader,
    val_loader,
    model,
    metrics: dict,
    wandb_logger,
    csv_logger,
):
    """Execute training loop."""
    print("\n🚀 Starting training...")
    
    # Create Lightning module
    lit_model = WSALightningModule(
        model=model,
        metrics=metrics,
        lr=config["optimizer"]["learning_rate"],
        batch_size=config["data"]["batch_size"],
    )
    
    # Create checkpoint directory
    checkpoint_dir = Path(config["checkpoint"]["save_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="wsa_best_{val_loss:.4f}",
        monitor=config["checkpoint"]["monitor"],
        mode=config["checkpoint"]["mode"],
        save_top_k=config["checkpoint"]["save_top_k"],
        verbose=True,
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config["optimizer"]["max_epochs"],
        accelerator=config["device"],
        devices="auto",
        logger=[wandb_logger, csv_logger],
        callbacks=[checkpoint_callback],
        log_every_n_steps=config["logging"]["log_every_n_steps"],
        num_sanity_val_steps=0,  # Skip sanity check
    )
    
    # Fit model
    trainer.fit(lit_model, train_loader, val_loader)
    
    print("\n✅ Training completed!")
    print(f"📦 Best model saved to: {checkpoint_dir}/wsa_best_*.ckpt")
    
    return trainer, lit_model


# ============================================================================
# Main
# ============================================================================

def main():
    """Main training script."""
    print("=" * 80)
    print("WSA Map Prediction Training Script")
    print("=" * 80)
    
    # Parse arguments
    args = parse_args()
    
    # Setup environment
    setup_environment(args.cuda_device)
    
    # Load configuration
    config = load_config(args.config)
    
    # Build scalers
    scalers = build_scalers(info=config["data"]["scalers"])
    
    # Create datasets and dataloaders
    train_dataset, val_dataset, test_dataset = create_datasets(config, scalers)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_data_workers"],
    )
    
    # Load pre-trained Surya encoder
    encoder = load_surya_encoder(config["model"]["checkpoint_path"])
    
    # Override freeze_encoder if CLI flag is set
    freeze_encoder = config["model"]["freeze_encoder"]
    if args.unfreeze_encoder:
        freeze_encoder = False
        print("🔓 Encoder will be trainable (--unfreeze-encoder)")
    
    # Create WSA model
    wsa_model = create_wsa_model(
        encoder=encoder,
        encoder_out_channels=config["model"]["embed_dim"],
        freeze_encoder=freeze_encoder,
    )
    
    # Create metrics
    train_loss_metrics, train_eval_metrics, val_eval_metrics = create_metrics()
    metrics = {
        "train_loss": train_loss_metrics,
        "train_metrics": train_eval_metrics,
        "val_metrics": val_eval_metrics,
    }
    
    # Setup loggers
    wandb_logger, csv_logger = setup_loggers(config)
    
    # Train
    trainer, lit_model = train(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        model=wsa_model,
        metrics=metrics,
        wandb_logger=wandb_logger,
        csv_logger=csv_logger,
    )
    
    print("\n" + "=" * 80)
    print("Training Summary")
    print("=" * 80)
    print(f"Job ID: {config['job_id']}")
    print(f"Max Epochs: {config['optimizer']['max_epochs']}")
    print(f"Learning Rate: {config['optimizer']['learning_rate']}")
    print(f"Batch Size: {config['data']['batch_size']}")
    print(f"Encoder Frozen: {config['model']['freeze_encoder']}")
    print(f"Train CRs: {config['data']['train_crs']}")
    print(f"Val CRs: {config['data']['val_crs']}")
    print(f"Test CRs: {config['data']['test_crs']}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()