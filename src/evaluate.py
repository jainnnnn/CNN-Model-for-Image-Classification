
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from data_loader import test_ds, class_names

# Load best trained model
model = tf.keras.models.load_model(
    "models/best_model.keras"
)

# Evaluate test dataset
test_loss, test_accuracy = model.evaluate(test_ds)

print("\nTest Accuracy:", round(test_accuracy * 100, 2), "%")
print("Test Loss:", round(test_loss, 4))

# Actual labels
y_true = np.concatenate([
    labels.numpy() for images, labels in test_ds
])

# Predicted labels
predictions = model.predict(test_ds)
y_pred = np.argmax(predictions, axis=1)

# Classification report
print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=class_names
))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(9, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("CNN Confusion Matrix")

os.makedirs("outputs/plots", exist_ok=True)

plt.savefig("outputs/plots/confusion_matrix.png")
plt.show()
