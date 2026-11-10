import tensorflow as tf
import os
import matplotlib.pyplot as plt
from data_loader import train_ds, val_ds, class_names
from model import build_cnn

# Create folders for saving results
os.makedirs("models", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)

# Build CNN
model = build_cnn()
model.summary()

# Callbacks for early stopping and saving the best model
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", 
    patience=5, 
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "models/best_model.keras", 
    monitor="val_accuracy", 
    save_best_only=True, 
    mode="max"
)

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=[early_stopping, checkpoint]
)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.savefig('outputs/plots/training_history.png')
print("\nTraining complete! Model and plots saved successfully.")