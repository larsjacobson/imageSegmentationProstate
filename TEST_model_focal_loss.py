import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from skimage.transform import resize
from skimage.io import imread, imsave
import os
import matplotlib.pyplot as plt

def focal_loss_fixed(y_true, y_pred, gamma=2., alpha=0.25):
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -tf.reduce_sum(alpha * tf.pow(1. - pt_1, gamma) * tf.math.log(pt_1)) - \
           tf.reduce_sum((1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1. - pt_0))#

# Define a function to convert class indices back to colors for visualization
def class_to_color(mask, colors):
    colored_mask = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        colored_mask[mask == i] = color
    return colored_mask

# Define the colors corresponding to each class
colors = [
    [0, 0, 0],       # Class 0: Background
    [0, 255, 0],     # Class 1: Green
    [0, 0, 255],     # Class 2: Blue
    [255, 0, 0]      # Class 3: Red
]

def load_test_images(test_image_dir):
    """Function to load and preprocess test images."""
    test_images = []
    original_images = []
    filenames = []

    for filename in os.listdir(test_image_dir):
        if filename.endswith('.png'):
            img_path = os.path.join(test_image_dir, filename)
            img = imread(img_path)
            original_images.append(img)
            img_gray = imread(img_path, as_gray=True)
            img_gray = img_gray.astype(np.float32)
            img_gray /= np.max(img_gray)  # Normalize to [0, 1]
            test_images.append(img_gray)
            filenames.append(filename)

    return np.array(test_images).reshape(-1, 128, 128, 1), original_images, filenames

# Load the trained model
model = load_model('prostate_segmentation_model_focal_loss.h5', custom_objects={'focal_loss_fixed': focal_loss_fixed})

# Load test images
test_image_dir = 'test/image'
X_test, original_images, test_filenames = load_test_images(test_image_dir)

# Resize test images if necessary
image_size = 128
X_test_resized = np.array([resize(image, (image_size, image_size, 1)) for image in X_test])

# Predict masks for test data
predicted_masks = model.predict(X_test_resized)

# Specify directory to save the predicted masks and combined images
save_mask_dir = 'test/mask/model3'
save_combined_dir = 'test/combined'
os.makedirs(save_mask_dir, exist_ok=True)
os.makedirs(save_combined_dir, exist_ok=True)

# Visualize and save the predicted masks and combined images
for i, (pred_mask, original_image, filename) in enumerate(zip(predicted_masks, original_images, test_filenames)):
    # Convert probabilities to class labels
    pred_label = np.argmax(pred_mask, axis=-1)

    # Convert class labels to colors for visualization
    colored_mask = class_to_color(pred_label, colors)

    # Save the colored mask
    plt.imsave(os.path.join(save_mask_dir, f"predicted_mask_{filename}"), colored_mask)

    # Combine original image with the colored mask
    plt.figure(figsize=(8, 8))
    plt.imshow(original_image)
    plt.imshow(colored_mask, alpha=0.5)  # Overlay mask with transparency
    plt.axis('off')
    plt.savefig(os.path.join(save_combined_dir, f"combined_{filename}"), bbox_inches='tight')
    plt.close()

# Add any additional post-processing if needed