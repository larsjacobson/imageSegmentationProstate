import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from skimage.transform import resize
from skimage.io import imread
import os
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('prostate_segmentation_model_single_class.h5')

# Function to load and preprocess test images
def load_test_images(test_image_dir):
    """Load and preprocess test images."""
    test_images = []
    filenames = []

    for filename in os.listdir(test_image_dir):


        if filename.endswith('.png'):
            img_path = os.path.join(test_image_dir, filename)
            img = imread(img_path, as_gray=True)
            img = img.astype(np.float32)
            img /= np.max(img)  # Normalize to [0, 1]
            test_images.append(img)
            filenames.append(filename)

    return np.array(test_images), filenames

# Load test images
test_image_dir = 'test/image'
X_test, test_filenames = load_test_images(test_image_dir)

# Resize test images if necessary
image_size = 128
X_test_resized = np.array([resize(image, (image_size, image_size, 1)) for image in X_test])

# Predict masks for test data
predicted_masks = model.predict(X_test_resized)

# Specify directory to save the predicted masks
save_dir = 'test/mask'

# Create the directory if it does not exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Visualize, save the predictions, and handle a large number of images
for i, pred_mask in enumerate(predicted_masks):
    plt.figure(figsize=(8, 8))
    plt.imshow(pred_mask.squeeze(), cmap='gray')
    #plt.title(f"Prediction for {test_filenames[i]}")
    plt.axis('off')

    # Save the figure
    plt.savefig(os.path.join(save_dir, f"predicted_mask_{test_filenames[i]}"), bbox_inches='tight')
    plt.close()  # Close the figure to free memory

# Optionally, add more sophisticated visualization or post-processing