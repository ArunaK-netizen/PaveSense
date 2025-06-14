# Training configuration
class TrainingConfig:
    # Model parameters
    SEQUENCE_LENGTH = 50
    N_FEATURES = 6
    N_CLASSES = 2

    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    EPOCHS = 100

    # Early stopping
    EARLY_STOPPING_PATIENCE = 15

    # Data augmentation
    USE_DATA_AUGMENTATION = False

    # Model architecture
    CNN_FILTERS = [64, 128, 64]
    LSTM_HIDDEN = 100
    LSTM_LAYERS = 2
    DROPOUT_RATE = 0.3
