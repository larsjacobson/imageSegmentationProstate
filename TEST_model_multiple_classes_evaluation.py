import numpy as np
import tensorflow as tf
from scipy import ndimage
from tensorflow.keras.models import load_model
from skimage.transform import resize
from skimage.io import imread
import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt

# Load the trained model
model = load_model('prostate_segmentation_model_multiple_classes_old.h5')

# Function to load and preprocess test images
def load_test_images(test_image_dir, target_size=(128, 128)):
    test_images = []
    filenames = []

    for filename in os.listdir(test_image_dir):
        if filename.endswith('.png'):
            img_path = os.path.join(test_image_dir, filename)
            img = imread(img_path, as_gray=True)
            img_resized = resize(img, target_size)
            test_images.append(img_resized)
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
save_dir = 'test/mask/model4'

# Create the directory if it does not exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Initialize lists to store evaluation results
dice_coeffs = []
rvds = []
hausdorff_dists = []
average_surface_dists = []

# Define evaluation functions
def dice_similarity_coefficient(mask1, mask2):
    intersection = np.sum(mask1 * mask2)
    union = np.sum(mask1) + np.sum(mask2)
    return (2.0 * intersection) / union

def relative_volume_difference(mask1, mask2):
    vol_mask1 = np.sum(mask1)
    vol_mask2 = np.sum(mask2)
    return abs(vol_mask1 - vol_mask2) / max(vol_mask1, vol_mask2)

def hausdorff_distance(mask1, mask2):
    mask1 = mask1.squeeze()
    mask2 = mask2.squeeze()
    hausdorff_1_to_2 = directed_hausdorff(mask1, mask2)[0]
    hausdorff_2_to_1 = directed_hausdorff(mask2, mask1)[0]
    return max(hausdorff_1_to_2, hausdorff_2_to_1)

def average_surface_distance(mask1, mask2, pixel_spacing=1.0):
    mask1_flat = mask1.squeeze()
    mask2_flat = mask2.squeeze()
    dist1 = distance_transform_edt(mask1_flat, sampling=pixel_spacing)
    dist2 = distance_transform_edt(mask2_flat, sampling=pixel_spacing)
    asd_1_to_2 = np.mean(dist1[mask2_flat > 0])
    asd_2_to_1 = np.mean(dist2[mask1_flat > 0])
    return (asd_1_to_2 + asd_2_to_1) / 2

# Visualize, save the predictions, and evaluate metrics
for i, pred_mask in enumerate(predicted_masks):
    # Load ground truth mask
    ground_truth_mask = imread(os.path.join('test', 'image', test_filenames[i]), as_gray=True)

    # Resize ground truth mask if necessary
    ground_truth_mask_resized = resize(ground_truth_mask, (image_size, image_size), anti_aliasing=True)

    # Squeeze the predicted mask to remove the extra dimension
    pred_mask = np.squeeze(pred_mask)

    # Handle multi-channel masks
    if pred_mask.shape[-1] > 1:
        pred_mask = pred_mask[..., 0]  # Assuming the correct label is in channel 0

    # Calculate Dice Similarity Coefficient
    dice_coeff = dice_similarity_coefficient(ground_truth_mask_resized, pred_mask)
    dice_coeffs.append(dice_coeff)

    # Calculate Relative Volume Difference
    rvd = relative_volume_difference(ground_truth_mask_resized, pred_mask)
    rvds.append(rvd)

    # Calculate Hausdorff Distance
    hausdorff_dist = hausdorff_distance(ground_truth_mask_resized, pred_mask)
    hausdorff_dists.append(hausdorff_dist)

    # Calculate Average Surface Distance
    average_surface_dist = average_surface_distance(ground_truth_mask_resized, pred_mask)
    average_surface_dists.append(average_surface_dist)

    # Visualize and save the predicted mask
    plt.figure(figsize=(8, 8))
    plt.imshow(pred_mask, cmap='gray')
    plt.title(f"Prediction for {test_filenames[i]}")
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, f"predicted_mask_{test_filenames[i]}"), bbox_inches='tight')
    plt.close()  # Close the figure to free memory

# Calculate mean for all metrics
mean_dice_coeff = np.mean(dice_coeffs)
mean_rvd = np.mean(rvds)
mean_hausdorff_dist = np.mean(hausdorff_dists)
mean_average_surface_dist = np.mean(average_surface_dists)

# Calculate average for all metrics
average_dice_coeff = np.mean(dice_coeffs)
average_rvd = np.mean(rvds)
average_hausdorff_dist = np.mean(hausdorff_dists)
average_average_surface_dist = np.mean(average_surface_dists)

# Calculate 95th percentile for all metrics
percentile_95_dice_coeff = np.percentile(dice_coeffs, 95)
percentile_95_rvd = np.percentile(rvds, 95)
percentile_95_hausdorff_dist = np.percentile(hausdorff_dists, 95)
percentile_95_average_surface_dist = np.percentile(average_surface_dists, 95)

# Print results
print("Evaluation Results:")
print(f"Mean Dice Similarity Coefficient: {mean_dice_coeff}")
print(f"Mean Relative Volume Difference: {mean_rvd}")
print(f"Mean Hausdorff Distance: {mean_hausdorff_dist}")
print(f"Mean Average Surface Distance: {mean_average_surface_dist}")
print()
print("Average Results:")
print(f"Average Dice Similarity Coefficient: {average_dice_coeff}")
print(f"Average Relative Volume Difference: {average_rvd}")
print(f"Average Hausdorff Distance: {average_hausdorff_dist}")
print(f"Average Average Surface Distance: {average_average_surface_dist}")
print()
print("95th Percentile Results:")
print(f"95th Percentile Dice Similarity Coefficient: {percentile_95_dice_coeff}")
print(f"95th Percentile Relative Volume Difference: {percentile_95_rvd}")
print(f"95th Percentile Hausdorff Distance: {percentile_95_hausdorff_dist}")
print(f"95th Percentile Average Surface Distance: {percentile_95_average_surface_dist}")