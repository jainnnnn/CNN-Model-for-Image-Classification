
import matplotlib.pyplot as plt
import os
import random
from PIL import Image

dataset_path = "dataset/seg_train"

classes = sorted(os.listdir(dataset_path))

plt.figure(figsize=(12, 7))

for i, category in enumerate(classes):

    folder_path = os.path.join(dataset_path, category)

    if not os.path.isdir(folder_path):
        continue

    images = os.listdir(folder_path)

    # Select one random image
    image_name = random.choice(images)

    image_path = os.path.join(folder_path, image_name)

    # Open image
    image = Image.open(image_path)

    # Display image
    plt.subplot(2, 3, i + 1)
    plt.imshow(image)
    plt.title(category)
    plt.axis("off")

plt.tight_layout()
plt.show()
