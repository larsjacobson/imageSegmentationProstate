import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from skimage.transform import resize
from skimage.io import imread
import os
import matplotlib.pyplot as plt
import cv2

# Load the trained model
model = load_model('prostate_segmentation_model_single_class.h5')

# Function to load and preprocess test images
def load_test_images(test_image_dir):
    test_images = []
    original_images = []
    filenames = []

    for filename in os.listdir(test_image_dir):
        if filename.endswith('.png'):
            img_path = os.path.join(test_image_dir, filename)
            img = imread(img_path, as_gray=True)
            original_images.append(img)
            img_resized = resize(img, (128, 128), anti_aliasing=True)
            test_images.append(img_resized)
            filenames.append(filename)

    return np.array(test_images).reshape(-1, 128, 128, 1), original_images, filenames

# Load test images
test_image_dir = 'test/image'
X_test, original_images, test_filenames = load_test_images(test_image_dir)

# Predict masks
predicted_masks = model.predict(X_test)

# Overlay, visualize, and save the predictions
save_dir = 'test/combined'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for i, (pred_mask, original_image, filename) in enumerate(zip(predicted_masks, original_images, test_filenames)):
    # Resize mask back to original image size
    mask_resized = resize(pred_mask.squeeze(), original_image.shape, anti_aliasing=True)

    # Overlay mask on the original image
    fig_combined, ax = plt.subplots()
    ax.imshow(original_image, cmap='gray')
    ax.imshow(mask_resized, cmap='Reds', alpha=0.9)  # Overlay mask with transparency
    ax.axis('off')

    # Construct combined filename and save path
    combined_filename = f"{filename.split('.')[0]}_combined.png"
    combined_path = os.path.join(save_dir, combined_filename)

    # Save the combined image
    fig_combined.savefig(combined_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig_combined)