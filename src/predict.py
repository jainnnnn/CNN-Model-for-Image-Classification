
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import os

from data_loader import test_ds, class_names

# Load trained model
model = tf.keras.models.load_model(
    "models/best_model.keras"
)

# Collect test images and labels
images_list = []
labels_list = []

for images, labels in test_ds:
    images_list.extend(images.numpy())
    labels_list.extend(labels.numpy())

# Select 10 random images
indices = random.sample(
    range(len(images_list)), 10
)

plt.figure(figsize=(15, 10))

for i, index in enumerate(indices):

    image = images_list[index]
    actual_label = labels_list[index]

    # Predict image category
    prediction = model.predict(
        np.expand_dims(image, axis=0),
        verbose=0
    )

    predicted_label = np.argmax(prediction)

    plt.subplot(2, 5, i + 1)
    plt.imshow(image.astype("uint8"))

    actual = class_names[actual_label]
    predicted = class_names[predicted_label]

    plt.title(
        f"Actual: {actual}\nPredicted: {predicted}"
    )

    plt.axis("off")

plt.tight_layout()

os.makedirs("outputs/plots", exist_ok=True)

plt.savefig("outputs/plots/predictions.png")
plt.show()
