import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


class DatasetProcessor:
    def __init__(self, sequence_length=50):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()

    def load_all_datasets(self, dataset_dir="datasets"):
        """Load and combine all collected datasets"""
        csv_files = glob.glob(os.path.join(dataset_dir, "pothole_data_*.csv"))

        if not csv_files:
            raise ValueError("No dataset files found! Run data_collector.py first.")

        print(f"📂 Found {len(csv_files)} dataset files")

        all_data = []
        for file in csv_files:
            df = pd.read_csv(file)
            df['session'] = os.path.basename(file)
            all_data.append(df)
            print(f"   {os.path.basename(file)}: {len(df)} samples")

        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"📊 Combined dataset: {len(combined_df)} total samples")

        return combined_df

    def analyze_dataset(self, df):
        """Analyze the collected dataset"""
        print("\n📈 Dataset Analysis")
        print("=" * 40)

        # Basic statistics
        print(f"Total samples: {len(df)}")
        print(f"Pothole samples: {sum(df['label'])} ({sum(df['label']) / len(df) * 100:.1f}%)")
        print(f"Normal samples: {len(df) - sum(df['label'])} ({(len(df) - sum(df['label'])) / len(df) * 100:.1f}%)")

        # Duration
        duration = (df['timestamp'].max() - df['timestamp'].min()) / 60
        print(f"Total duration: {duration:.1f} minutes")

        # Sensor statistics
        print(f"\nSensor Statistics:")
        for col in ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']:
            print(
                f"  {col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}, range=[{df[col].min():.2f}, {df[col].max():.2f}]")

        # Plot distributions
        self.plot_sensor_distributions(df)

    def plot_sensor_distributions(self, df):
        """Plot sensor data distributions"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        sensors = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']

        for i, sensor in enumerate(sensors):
            row, col = i // 3, i % 3
            ax = axes[row, col]

            # Plot distributions for normal vs pothole
            normal_data = df[df['label'] == 0][sensor]
            pothole_data = df[df['label'] == 1][sensor]

            ax.hist(normal_data, bins=50, alpha=0.7, label='Normal', density=True)
            ax.hist(pothole_data, bins=50, alpha=0.7, label='Pothole', density=True)
            ax.set_title(f'{sensor} Distribution')
            ax.legend()

        plt.tight_layout()
        plt.savefig('datasets/sensor_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("📊 Sensor distributions saved to 'datasets/sensor_distributions.png'")

    def create_sequences(self, df):
        """Create sequences for CNN-LSTM training"""
        print(f"\n🔄 Creating sequences (length={self.sequence_length})...")

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Extract sensor features
        feature_columns = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
        sensor_data = df[feature_columns].values
        labels = df['label'].values

        # Normalize sensor data
        sensor_data_normalized = self.scaler.fit_transform(sensor_data)

        # Create sequences
        sequences = []
        sequence_labels = []

        for i in range(len(sensor_data_normalized) - self.sequence_length + 1):
            sequence = sensor_data_normalized[i:i + self.sequence_length]
            # Use majority vote for sequence label
            label_window = labels[i:i + self.sequence_length]
            sequence_label = 1 if np.mean(label_window) > 0.5 else 0

            sequences.append(sequence)
            sequence_labels.append(sequence_label)

        sequences = np.array(sequences)
        sequence_labels = np.array(sequence_labels)

        print(f"✅ Created {len(sequences)} sequences")
        print(
            f"   Pothole sequences: {sum(sequence_labels)} ({sum(sequence_labels) / len(sequence_labels) * 100:.1f}%)")
        print(
            f"   Normal sequences: {len(sequence_labels) - sum(sequence_labels)} ({(len(sequence_labels) - sum(sequence_labels)) / len(sequence_labels) * 100:.1f}%)")

        return sequences, sequence_labels

    def balance_dataset(self, sequences, labels):
        """Balance the dataset by undersampling majority class"""
        normal_indices = np.where(labels == 0)[0]
        pothole_indices = np.where(labels == 1)[0]

        print(f"\n⚖️  Balancing dataset...")
        print(f"   Normal sequences: {len(normal_indices)}")
        print(f"   Pothole sequences: {len(pothole_indices)}")

        # Undersample normal sequences to match pothole sequences
        if len(normal_indices) > len(pothole_indices):
            sampled_normal_indices = np.random.choice(
                normal_indices,
                size=len(pothole_indices),
                replace=False
            )
            balanced_indices = np.concatenate([sampled_normal_indices, pothole_indices])
        else:
            balanced_indices = np.concatenate([normal_indices, pothole_indices])

        # Shuffle indices
        np.random.shuffle(balanced_indices)

        balanced_sequences = sequences[balanced_indices]
        balanced_labels = labels[balanced_indices]

        print(f"✅ Balanced dataset: {len(balanced_sequences)} sequences")
        print(f"   Normal: {sum(balanced_labels == 0)} ({sum(balanced_labels == 0) / len(balanced_labels) * 100:.1f}%)")
        print(
            f"   Pothole: {sum(balanced_labels == 1)} ({sum(balanced_labels == 1) / len(balanced_labels) * 100:.1f}%)")

        return balanced_sequences, balanced_labels

    def split_dataset(self, sequences, labels):
        """Split dataset into train/validation/test sets"""
        print(f"\n📊 Splitting dataset...")

        # First split: train+val (80%) and test (20%)
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            sequences, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # Second split: train (64%) and val (16%)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval
        )

        print(f"✅ Dataset split completed:")
        print(f"   Training: {len(X_train)} sequences ({len(X_train) / len(sequences) * 100:.1f}%)")
        print(f"   Validation: {len(X_val)} sequences ({len(X_val) / len(sequences) * 100:.1f}%)")
        print(f"   Test: {len(X_test)} sequences ({len(X_test) / len(sequences) * 100:.1f}%)")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def save_processed_dataset(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Save processed dataset"""
        os.makedirs("datasets/processed", exist_ok=True)

        # Save arrays
        np.save("datasets/processed/X_train.npy", X_train)
        np.save("datasets/processed/X_val.npy", X_val)
        np.save("datasets/processed/X_test.npy", X_test)
        np.save("datasets/processed/y_train.npy", y_train)
        np.save("datasets/processed/y_val.npy", y_val)
        np.save("datasets/processed/y_test.npy", y_test)

        # Save scaler
        import joblib
        joblib.dump(self.scaler, "datasets/processed/scaler.pkl")

        # Save metadata
        metadata = {
            "sequence_length": self.sequence_length,
            "n_features": 6,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
            "train_pothole_ratio": float(np.mean(y_train)),
            "val_pothole_ratio": float(np.mean(y_val)),
            "test_pothole_ratio": float(np.mean(y_test))
        }

        import json
        with open("datasets/processed/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"💾 Processed dataset saved to 'datasets/processed/'")


def main():
    print("🔄 Processing Collected Dataset")
    print("=" * 40)

    processor = DatasetProcessor(sequence_length=50)

    # Load datasets
    df = processor.load_all_datasets()

    # Analyze dataset
    processor.analyze_dataset(df)

    # Create sequences
    sequences, labels = processor.create_sequences(df)

    # Balance dataset
    balanced_sequences, balanced_labels = processor.balance_dataset(sequences, labels)

    # Split dataset
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split_dataset(
        balanced_sequences, balanced_labels
    )

    # Save processed dataset
    processor.save_processed_dataset(X_train, X_val, X_test, y_train, y_val, y_test)

    print("\n✅ Dataset processing completed!")
    print("📁 Ready for training with 'train_real_model.py'")


if __name__ == "__main__":
    main()
