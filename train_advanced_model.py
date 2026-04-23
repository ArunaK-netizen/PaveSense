"""
PaveSense Advanced Model Training
====================================
Trains the multi-class PaveSense model on synthetic + real data.

Usage:
    python train_advanced_model.py

Output:
    models/pavesense_advanced.pth       — PyTorch checkpoint
    models/pavesense_advanced.onnx      — ONNX model for mobile
    models/feature_normalizer.npz       — Feature normalizer stats
    models/training_metrics.png         — Training curves
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
import os
import json
import time
from datetime import datetime

from models.advanced_model import PaveSenseModel
from models.feature_engineering import compute_features_batch, FeatureNormalizer
from models.synthetic_data import generate_dataset, augment_dataset


# ==========================================
# Configuration
# ==========================================
class TrainConfig:
    # Data
    N_SAMPLES_PER_CLASS = 3000
    AUGMENTATION_FACTOR = 2
    SEQUENCE_LENGTH = 100
    N_RAW_FEATURES = 6
    N_ENGINEERED_FEATURES = 13
    N_CLASSES = 5

    # Model architecture
    CNN_CHANNELS = 64
    LSTM_HIDDEN = 128
    LSTM_LAYERS = 2
    NUM_ATTENTION_HEADS = 4
    DROPOUT = 0.3

    # Training
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 80
    EARLY_STOPPING_PATIENCE = 15

    # Splits
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    # Paths
    MODEL_DIR = 'models'
    CHECKPOINT_PATH = 'models/pavesense_advanced.pth'
    ONNX_PATH = 'models/pavesense_advanced.onnx'
    NORMALIZER_PATH = 'models/feature_normalizer.npz'

    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    CLASS_NAMES = ['normal', 'pothole', 'speed_bump', 'phone_drop', 'disturbance']


def train():
    config = TrainConfig()
    print("=" * 70)
    print("🚗 PaveSense Advanced Model Training")
    print("=" * 70)
    print(f"🖥️  Device: {config.DEVICE}")
    print(f"📊 Classes: {config.CLASS_NAMES}")
    print(f"🔢 Samples per class: {config.N_SAMPLES_PER_CLASS}")
    print(f"📐 Sequence length: {config.SEQUENCE_LENGTH}")
    print(f"🔧 Features: {config.N_RAW_FEATURES} raw → {config.N_ENGINEERED_FEATURES} engineered")
    print()

    # ==========================================
    # Step 1: Generate Synthetic Data
    # ==========================================
    print("📦 Step 1: Generating synthetic data...")
    raw_data, labels = generate_dataset(
        n_samples_per_class=config.N_SAMPLES_PER_CLASS,
        sequence_length=config.SEQUENCE_LENGTH,
        seed=42,
    )
    print(f"   Generated {len(labels)} base samples")
    print(f"   Distribution: {dict(zip(config.CLASS_NAMES, np.bincount(labels)))}")

    # Augment
    print(f"   Augmenting {config.AUGMENTATION_FACTOR}x...")
    raw_data, labels = augment_dataset(raw_data, labels,
                                        augmentation_factor=config.AUGMENTATION_FACTOR, seed=123)
    print(f"   After augmentation: {len(labels)} samples")

    # ==========================================
    # Step 2: Feature Engineering
    # ==========================================
    print("\n🔬 Step 2: Computing engineered features...")
    t0 = time.time()
    features = compute_features_batch(raw_data, sampling_rate=50)
    print(f"   Feature shape: {features.shape}")
    print(f"   Computed in {time.time() - t0:.1f}s")

    # ==========================================
    # Step 3: Normalize
    # ==========================================
    print("\n📏 Step 3: Normalizing features...")
    normalizer = FeatureNormalizer(n_features=config.N_ENGINEERED_FEATURES)
    features_flat = features.reshape(-1, config.N_ENGINEERED_FEATURES)
    normalizer.fit(features_flat)
    features_normalized = normalizer.transform(features.reshape(-1, config.N_ENGINEERED_FEATURES))
    features_normalized = features_normalized.reshape(features.shape)
    normalizer.save(config.NORMALIZER_PATH)
    print(f"   Normalizer saved to {config.NORMALIZER_PATH}")

    # ==========================================
    # Step 4: Train/Val/Test Split
    # ==========================================
    print("\n✂️  Step 4: Splitting data...")
    n = len(labels)
    idx = np.random.RandomState(42).permutation(n)
    n_test = int(n * config.TEST_RATIO)
    n_val = int(n * config.VAL_RATIO)
    n_train = n - n_test - n_val

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    X_train = torch.FloatTensor(features_normalized[train_idx])
    y_train = torch.LongTensor(labels[train_idx])
    X_val = torch.FloatTensor(features_normalized[val_idx])
    y_val = torch.LongTensor(labels[val_idx])
    X_test = torch.FloatTensor(features_normalized[test_idx])
    y_test = torch.LongTensor(labels[test_idx])

    print(f"   Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Weighted sampler for class balance
    class_counts = np.bincount(labels[train_idx])
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels[train_idx]]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                              sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
                             shuffle=False, num_workers=0, pin_memory=True)

    # ==========================================
    # Step 5: Create Model
    # ==========================================
    print("\n🤖 Step 5: Creating model...")
    model = PaveSenseModel(
        sequence_length=config.SEQUENCE_LENGTH,
        n_features=config.N_ENGINEERED_FEATURES,
        n_classes=config.N_CLASSES,
        cnn_channels=config.CNN_CHANNELS,
        lstm_hidden=config.LSTM_HIDDEN,
        lstm_layers=config.LSTM_LAYERS,
        num_attention_heads=config.NUM_ATTENTION_HEADS,
        dropout=config.DROPOUT,
    ).to(config.DEVICE)

    total_params, trainable_params = PaveSenseModel.count_parameters(model)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # Loss with class weights (emphasize pothole class)
    class_weight_tensor = torch.FloatTensor([1.0, 2.0, 1.5, 1.5, 1.0]).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)

    optimizer = optim.AdamW(model.parameters(),
                            lr=config.LEARNING_RATE,
                            weight_decay=config.WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE,
        epochs=config.EPOCHS,
        steps_per_epoch=len(train_loader),
    )

    # ==========================================
    # Step 6: Training Loop
    # ==========================================
    print(f"\n🚀 Step 6: Training for {config.EPOCHS} epochs...")
    print("-" * 70)

    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(config.EPOCHS):
        # --- Train ---
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(config.DEVICE)
            batch_y = batch_y.to(config.DEVICE)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=-1)
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_x.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # --- Validate ---
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(config.DEVICE)
                batch_y = batch_y.to(config.DEVICE)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                preds = torch.argmax(logits, dim=-1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_x.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print progress
        lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch + 1:3d}/{config.EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {lr:.6f}")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'model_config': {
                    'sequence_length': config.SEQUENCE_LENGTH,
                    'n_features': config.N_ENGINEERED_FEATURES,
                    'n_classes': config.N_CLASSES,
                    'cnn_channels': config.CNN_CHANNELS,
                    'lstm_hidden': config.LSTM_HIDDEN,
                    'lstm_layers': config.LSTM_LAYERS,
                    'num_attention_heads': config.NUM_ATTENTION_HEADS,
                    'dropout': config.DROPOUT,
                },
                'class_names': config.CLASS_NAMES,
                'normalizer_path': config.NORMALIZER_PATH,
            }
            torch.save(checkpoint, config.CHECKPOINT_PATH)
            print(f"  ✅ Best model saved! (val_acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"\n  ⏱️  Early stopping at epoch {epoch + 1} (patience={config.EARLY_STOPPING_PATIENCE})")
                break

    print("-" * 70)
    print(f"✅ Training complete! Best val accuracy: {best_val_acc:.4f}")

    # ==========================================
    # Step 7: Test Evaluation
    # ==========================================
    print(f"\n📊 Step 7: Evaluating on test set...")

    # Load best model
    checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(config.DEVICE)
            logits = model(batch_x)
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Overall accuracy
    test_acc = np.mean(all_preds == all_labels)
    print(f"   Overall Test Accuracy: {test_acc:.4f} ({test_acc * 100:.1f}%)")

    # Per-class metrics
    print(f"\n   {'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"   {'-' * 55}")

    for i, name in enumerate(config.CLASS_NAMES):
        tp = np.sum((all_preds == i) & (all_labels == i))
        fp = np.sum((all_preds == i) & (all_labels != i))
        fn = np.sum((all_preds != i) & (all_labels == i))
        support = np.sum(all_labels == i)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"   {name:<15} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>10d}")

    # Confusion matrix
    print(f"\n   Confusion Matrix:")
    print(f"   {'':>15}", end='')
    for name in config.CLASS_NAMES:
        print(f" {name[:8]:>8}", end='')
    print()

    for i, name in enumerate(config.CLASS_NAMES):
        print(f"   {name:<15}", end='')
        for j in range(config.N_CLASSES):
            count = np.sum((all_labels == i) & (all_preds == j))
            print(f" {count:>8d}", end='')
        print()

    # ==========================================
    # Step 8: Export to ONNX
    # ==========================================
    print(f"\n📦 Step 8: Exporting to ONNX...")
    export_onnx(model, config)

    # ==========================================
    # Step 9: Save Training Plots
    # ==========================================
    try:
        save_training_plots(history, config)
        print(f"\n📊 Training plots saved to models/training_metrics.png")
    except Exception as e:
        print(f"⚠️ Could not save plots: {e}")

    print("\n" + "=" * 70)
    print(f"🎉 Training pipeline complete!")
    print(f"   📁 Model checkpoint: {config.CHECKPOINT_PATH}")
    print(f"   📁 ONNX model: {config.ONNX_PATH}")
    print(f"   📁 Normalizer: {config.NORMALIZER_PATH}")
    print(f"   📊 Test accuracy: {test_acc:.4f} ({test_acc * 100:.1f}%)")
    print("=" * 70)


def export_onnx(model, config):
    """Export trained model to ONNX format for mobile/edge inference."""
    model.eval()
    model = model.to('cpu')

    dummy_input = torch.randn(1, config.SEQUENCE_LENGTH, config.N_ENGINEERED_FEATURES)

    torch.onnx.export(
        model,
        dummy_input,
        config.ONNX_PATH,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['sensor_input'],
        output_names=['class_logits'],
        dynamic_axes={
            'sensor_input': {0: 'batch_size'},
            'class_logits': {0: 'batch_size'},
        },
    )

    # Validate ONNX
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(config.ONNX_PATH)
        onnx_input = dummy_input.numpy()
        onnx_output = session.run(None, {'sensor_input': onnx_input})[0]

        with torch.no_grad():
            torch_output = model(dummy_input).numpy()

        diff = np.abs(onnx_output - torch_output).max()
        print(f"   ✅ ONNX export validated (max diff: {diff:.8f})")

        # Model size
        onnx_size = os.path.getsize(config.ONNX_PATH)
        print(f"   📏 ONNX model size: {onnx_size / 1024:.1f} KB")

    except ImportError:
        print("   ⚠️ onnxruntime not installed — skipping ONNX validation")
        print(f"   📁 ONNX model saved to {config.ONNX_PATH}")


def save_training_plots(history, config):
    """Save training loss/accuracy curves."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', color='#00F0FF')
    axes[0].plot(history['val_loss'], label='Val Loss', color='#FF3366')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Acc', color='#00F0FF')
    axes[1].plot(history['val_acc'], label='Val Acc', color='#FF3366')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(config.MODEL_DIR, 'training_metrics.png'), dpi=150)
    plt.close()


if __name__ == '__main__':
    train()
