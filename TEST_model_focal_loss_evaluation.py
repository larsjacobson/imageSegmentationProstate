import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from skimage.transform import resize
from skimage.io import imread, imsave
import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt

def focal_loss_fixed(y_true, y_pred, gamma=2., alpha=0.25):
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -tf.reduce_sum(alpha * tf.pow(1. - pt_1, gamma) * tf.math.log(pt_1)) - \
           tf.reduce_sum((1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1. - pt_0))#

def class_to_color(mask, colors):
    colored_mask = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        colored_mask[mask == i] = color
    return colored_mask

colors = [
    [0, 0, 0],       # Class 0: Background
    [0, 255, 0],     # Class 1: Green
    [0, 0, 255],     # Class 2: Blue
    [255, 0, 0]      # Class 3: Red
]

def load_test_images(test_image_dir, target_size=(128, 128)):
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
            img_resized = resize(img_gray, target_size)
            test_images.append(img_resized)
            filenames.append(filename)

    return np.array(test_images).reshape(-1, target_size[0], target_size[1], 1), original_images, filenames

model = load_model('prostate_segmentation_model_focal_loss.h5', custom_objects={'focal_loss_fixed': focal_loss_fixed})

test_image_dir = 'test/image'
X_test, original_images, test_filenames = load_test_images(test_image_dir)

image_size = 128
X_test_resized = np.array([resize(image, (image_size, image_size, 1)) for image in X_test])

predicted_masks = model.predict(X_test_resized)

save_mask_dir = 'test/mask/model3'
save_combined_dir = 'test/combined'
os.makedirs(save_mask_dir, exist_ok=True)
os.makedirs(save_combined_dir, exist_ok=True)

dice_coeffs = []
rvds = []
hausdorff_dists = []
average_surface_dists = []

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

for i, pred_mask in enumerate(predicted_masks):
    ground_truth_mask = imread(os.path.join('test', 'image', test_filenames[i]), as_gray=True)
    ground_truth_mask_resized = resize(ground_truth_mask, (image_size, image_size), anti_aliasing=True)
    pred_mask = np.squeeze(pred_mask)

    if pred_mask.shape[-1] > 1:
        pred_mask = pred_mask[..., 0]

    dice_coeff = dice_similarity_coefficient(ground_truth_mask_resized, pred_mask)
    dice_coeffs.append(dice_coeff)

    rvd = relative_volume_difference(ground_truth_mask_resized, pred_mask)
    rvds.append(rvd)

    hausdorff_dist = hausdorff_distance(ground_truth_mask_resized, pred_mask)
    hausdorff_dists.append(hausdorff_dist)

    average_surface_dist = average_surface_distance(ground_truth_mask_resized, pred_mask)
    average_surface_dists.append(average_surface_dist)

    plt.figure(figsize=(8, 8))
    plt.imshow(pred_mask, cmap='gray')
    plt.title(f"Prediction for {test_filenames[i]}")
    plt.axis('off')
    plt.savefig(os.path.join(save_mask_dir, f"predicted_mask_{test_filenames[i]}"), bbox_inches='tight')
    plt.close()

mean_dice_coeff = np.mean(dice_coeffs)
mean_rvd = np.mean(rvds)
mean_hausdorff_dist = np.mean(hausdorff_dists)
mean_average_surface_dist = np.mean(average_surface_dists)

average_dice_coeff = np.mean(dice_coeffs)
average_rvd = np.mean(rvds)
average_hausdorff_dist = np.mean(hausdorff_dists)
average_average_surface_dist = np.mean(average_surface_dists)

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