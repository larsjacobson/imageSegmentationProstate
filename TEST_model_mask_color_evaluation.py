from tensorflow.keras.models import load_model
from skimage.transform import resize
from skimage.io import imread, imsave
from skimage import img_as_ubyte
import numpy as np
import os
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt

# Load the trained model from the directory
model = load_model('prostate_segmentation_model_multiple_classes_final')

def prepare_image(img_path):
    img = imread(img_path)
    img = resize(img, (128, 128), anti_aliasing=True)
    img = img.astype(np.float32) / np.max(img)  # Normalize to [0, 1]

    if img.ndim == 2:  # If grayscale, convert to RGB
        img = np.stack((img,) * 3, axis=-1)
    elif img.shape[-1] == 4:  # Drop alpha channel if present
        img = img[..., :3]

    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_mask(model, img):
    pred_mask = model.predict(img)
    pred_mask = np.argmax(pred_mask, axis=-1)  # Class labels
    pred_mask = pred_mask[0]  # Remove batch dimension
    return pred_mask

def labels_to_colors(pred_mask):
    label_colors = {
        0: [0, 0, 0],   # Background
        1: [0, 128, 0],  # Green
        2: [0, 0, 255],  # Blue
        3: [255, 0, 0]   # Red
    }
    color_mask = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    for label, color in label_colors.items():
        color_mask[pred_mask == label] = color
    return color_mask

# Evaluation metrics functions
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

# Directories
combined_save_dir = 'test/combined'
mask_save_dir = 'test/mask'
test_img_dir = 'test/image'
ground_truth_dir = 'test/image'  # Ground truth masks directory

if not os.path.exists(combined_save_dir):
    os.makedirs(combined_save_dir)
if not os.path.exists(mask_save_dir):
    os.makedirs(mask_save_dir)

test_img_paths = [os.path.join(test_img_dir, fname) for fname in os.listdir(test_img_dir) if fname.endswith('.png')]

# Lists for storing metric results
dice_coeffs, rvds, hausdorff_dists, average_surface_dists = [], [], [], []

# Loop through all test images
for img_path in test_img_paths:
    test_img = prepare_image(img_path)
    predicted_mask = predict_mask(model, test_img)
    color_mask = labels_to_colors(predicted_mask)

    # Load the original image and resize it to match the predicted mask
    original_img = imread(img_path)
    original_img = resize(original_img, color_mask.shape[:2], anti_aliasing=True)
    if original_img.ndim == 2:
        original_img = np.stack((original_img,) * 3, axis=-1)
    original_img = original_img.astype(np.float32)
    color_mask = color_mask.astype(np.float32)

    combined_img = 0.6 * original_img + 0.4 * color_mask
    combined_img = np.clip(combined_img, -1, 1)
    combined_img = img_as_ubyte(combined_img)

    # Load ground truth mask for evaluation
    ground_truth_path = os.path.join(ground_truth_dir, os.path.basename(img_path))
    ground_truth_mask = imread(ground_truth_path)

    # Resize ground truth mask to match the predicted mask's shape
    ground_truth_mask = resize(ground_truth_mask, (128, 128), anti_aliasing=True)

    # Calculate evaluation metrics
    dice_coeff = dice_similarity_coefficient(ground_truth_mask, predicted_mask)
    rvd = relative_volume_difference(ground_truth_mask, predicted_mask)
    hausdorff_dist = hausdorff_distance(ground_truth_mask, predicted_mask)
    average_surface_dist = average_surface_distance(ground_truth_mask, predicted_mask)

    # Append results to lists
    dice_coeffs.append(dice_coeff)
    rvds.append(rvd)
    hausdorff_dists.append(hausdorff_dist)
    average_surface_dists.append(average_surface_dist)

# Calculate mean for all metrics
mean_dice_coeff = np.mean(dice_coeffs)
mean_rvd = np.mean(rvds)
mean_hausdorff_dist = np.mean(hausdorff_dists)
mean_average_surface_dist = np.mean(average_surface_dists)

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
print("95th Percentile Results:")
print(f"95th Percentile Dice Similarity Coefficient: {percentile_95_dice_coeff}")
print(f"95th Percentile Relative Volume Difference: {percentile_95_rvd}")
print(f"95th Percentile Hausdorff Distance: {percentile_95_hausdorff_dist}")
print(f"95th Percentile Average Surface Distance: {percentile_95_average_surface_dist}")
