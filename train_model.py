# ============================================================
# Plant Disease Detection - Model Training Script
# ============================================================
# This script trains a Convolutional Neural Network (CNN)
# to classify plant leaf images into 4 categories:
#   - Healthy
#   - Early Blight
#   - Late Blight
#   - Leaf Mold
#
# DATASET: PlantVillage (available on Kaggle)
# Download from: https://www.kaggle.com/datasets/emmarex/plantdisease
#
# FOLDER STRUCTURE EXPECTED:
#   dataset/
#     train/
#       Healthy/         (images)
#       Early_Blight/    (images)
#       Late_Blight/     (images)
#       Leaf_Mold/       (images)
#     val/
#       (same structure as train/)
#
# HOW TO RUN:
#   python train_model.py
#
# The model will be saved to: model/plant_disease_model.h5
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ----------------------------------------------------------
# Configuration — change these settings as needed
# ----------------------------------------------------------
IMAGE_SIZE = (128, 128)     # Input image dimensions
BATCH_SIZE = 32             # Images processed per training step
EPOCHS = 20                 # Max training rounds (EarlyStopping may stop earlier)
NUM_CLASSES = 4             # Healthy, Early Blight, Late Blight, Leaf Mold
DATASET_DIR = 'dataset'     # Path to your dataset folder
MODEL_SAVE_PATH = 'model/plant_disease_model.h5'

# ----------------------------------------------------------
# Step 1: Prepare Data Generators
# ----------------------------------------------------------
# ImageDataGenerator handles:
# - Loading images from folders
# - Resizing to 128x128
# - Normalizing pixel values
# - Data augmentation (artificial variety to prevent overfitting)

print("[STEP 1] Setting up data generators...")

# Training data: augmented to improve generalization
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,          # Normalize pixels to [0, 1]
    rotation_range=20,           # Random rotation ±20 degrees
    width_shift_range=0.2,       # Shift image left/right
    height_shift_range=0.2,      # Shift image up/down
    shear_range=0.2,             # Shearing transformation
    zoom_range=0.2,              # Random zoom
    horizontal_flip=True,        # Mirror images
    fill_mode='nearest'          # Fill gaps after transforms
)

# Validation data: only normalize, no augmentation
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load training images from folder structure
train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, 'train'),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'     # Multi-class classification
)

# Load validation images
val_generator = val_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, 'val'),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Print class mapping (important: must match CLASS_NAMES in app.py)
print(f"\n[INFO] Class mapping: {train_generator.class_indices}")
print("[INFO] Make sure CLASS_NAMES in app.py matches this order!\n")

# ----------------------------------------------------------
# Step 2: Build the CNN Model
# ----------------------------------------------------------
# Architecture: 3 Convolutional blocks + Dense classifier
# Each conv block: Conv2D → MaxPooling → BatchNorm → Dropout

print("[STEP 2] Building CNN model...")

model = models.Sequential([
    # Input layer
    layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),

    # ---- Convolutional Block 1 ----
    # Learn basic features: edges, colors, textures
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2)),       # Reduce spatial size by 2x
    layers.BatchNormalization(),                 # Normalize activations
    layers.Dropout(0.25),                        # Randomly drop 25% neurons

    # ---- Convolutional Block 2 ----
    # Learn more complex patterns
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.BatchNormalization(),
    layers.Dropout(0.25),

    # ---- Convolutional Block 3 ----
    # Learn high-level disease features
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.BatchNormalization(),
    layers.Dropout(0.4),

    # ---- Classifier Head ----
    layers.Flatten(),                            # Convert 2D features to 1D
    layers.Dense(256, activation='relu'),        # Fully connected layer
    layers.BatchNormalization(),
    layers.Dropout(0.5),                         # Strong dropout before output
    layers.Dense(NUM_CLASSES, activation='softmax')  # Output: probability per class
])

# Print model architecture summary
model.summary()

# ----------------------------------------------------------
# Step 3: Compile the Model
# ----------------------------------------------------------
print("\n[STEP 3] Compiling model...")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',    # Standard for multi-class problems
    metrics=['accuracy']
)

# ----------------------------------------------------------
# Step 4: Set Up Training Callbacks
# ----------------------------------------------------------
# Callbacks are functions called during training

callbacks = [
    # Stop training if validation accuracy doesn't improve for 5 epochs
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    # Save the best model during training
    ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# ----------------------------------------------------------
# Step 5: Train the Model
# ----------------------------------------------------------
print("\n[STEP 4] Starting training...\n")
os.makedirs('model', exist_ok=True)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# ----------------------------------------------------------
# Step 6: Plot Training Results
# ----------------------------------------------------------
print("\n[STEP 5] Saving training plots...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
ax1.plot(history.history['accuracy'], label='Train Accuracy', color='#22c55e', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Val Accuracy', color='#16a34a', linewidth=2, linestyle='--')
ax1.set_title('Model Accuracy Over Epochs', fontsize=14)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(alpha=0.3)

# Loss plot
ax2.plot(history.history['loss'], label='Train Loss', color='#ef4444', linewidth=2)
ax2.plot(history.history['val_loss'], label='Val Loss', color='#dc2626', linewidth=2, linestyle='--')
ax2.set_title('Model Loss Over Epochs', fontsize=14)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('model/training_history.png', dpi=150)
print("[INFO] Training plot saved to model/training_history.png")

# ----------------------------------------------------------
# Step 7: Final Evaluation
# ----------------------------------------------------------
print("\n[STEP 6] Evaluating on validation set...")
val_loss, val_accuracy = model.evaluate(val_generator, verbose=0)
print(f"\n{'='*40}")
print(f"  Final Validation Accuracy: {val_accuracy * 100:.2f}%")
print(f"  Final Validation Loss:     {val_loss:.4f}")
print(f"{'='*40}")
print(f"\n[SUCCESS] Model saved to: {MODEL_SAVE_PATH}")
print("[INFO] You can now run 'python app.py' to start the web server.")
