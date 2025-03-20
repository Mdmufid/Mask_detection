import os
import cv2
import numpy as np

# Define dataset directory
dataset_dir = "dataset"

# Define categories
categories = ["with_mask", "without_mask"]

for category in categories:
    path = os.path.join(dataset_dir, category)
    images = []
    
    # Check if directory exists
    if not os.path.exists(path):
        print(f"Error: {path} not found. Make sure to run collect_data.py first.")
        continue

    # Process images
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Skipping {img_name}, could not read file.")
            continue
        
        img = cv2.resize(img, (50, 50))  # Resize to 50x50
        images.append(img)

    # Convert list to numpy array
    images = np.array(images)

    # Save as .npy file
    np.save(f"{category}.npy", images)
    print(f"Saved {category}.npy with {len(images)} images.")
