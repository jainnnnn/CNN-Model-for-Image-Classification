
import os

# Location of our training dataset
DATASET_PATH = "dataset/seg_train"

# Six image categories
classes = os.listdir(DATASET_PATH)

print("CNN IMAGE CLASSIFICATION DATASET")
print("--------------------------------")

print("Number of classes:", len(classes))
print("\nClass names:")

for category in classes:
    folder_path = os.path.join(DATASET_PATH, category)

    if os.path.isdir(folder_path):
        images = os.listdir(folder_path)
        print(category, ":", len(images), "images")
