import numpy as np
import os
from skimage.io import imread
from pathlib import Path
from skimage.transform import resize
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.measure import label

def load_ground_truth_masks(ground_truth_dir, image_filenames):
    """
    Load ground truth masks from disk by adjusting the filenames.

    Args:
        ground_truth_dir: Path to the directory containing the ground truth masks.
        image_filenames: List of original image filenames, which will be modified to match ground truth mask filenames.

    Returns:
        A numpy array containing the ground truth masks.
    """
    ground_truth_masks = []
    for filename in image_filenames:
        # Replace "_image" with "_mask" in each filename to match the ground truth naming convention
        mask_filename = filename.replace('_image', '_mask')
        file_path = ground_truth_dir / mask_filename  # Construct the full path to the ground truth mask

        print(f"Loading ground truth file: {file_path}")  # Optional: print to verify correct file paths
        try:
            mask = imread(str(file_path))
            ground_truth_masks.append(mask)
        except FileNotFoundError as e:
            print(f"File not found: {file_path}")  # Print error message if file is not found
            raise e  # Optionally re-raise the exception to halt the script, or handle it as needed

    return np.array(ground_truth_masks)

def resize_mask(mask, output_shape):
    resized_mask = resize(mask, output_shape, preserve_range=True, anti_aliasing=True)
    return resized_mask


def dice_similarity_coefficient(mask1, mask2):
    # Resize mask2 to match mask1's shape, assuming mask1's shape is the target
    mask2_resized = resize(mask2, mask1.shape[:2], preserve_range=True, mode='constant', anti_aliasing=False)

    # Convert resized masks to boolean if they are not binary, assuming non-zero pixel values are foreground
    mask1_binary = mask1 > 0
    mask2_resized_binary = mask2_resized > 0

    intersection = np.sum(mask1_binary * mask2_resized_binary)
    union = np.sum(mask1_binary) + np.sum(mask2_resized_binary)
    return (2.0 * intersection) / union if union != 0 else 1.0


def relative_volume_difference(mask1, mask2):
    vol_mask1 = np.sum(mask1)
    vol_mask2 = np.sum(mask2)
    return abs(vol_mask1 - vol_mask2) / max(vol_mask1, vol_mask2)


def hausdorff_distance(mask1, mask2, resize_shape=None):
    # Optional: Resize masks to have the same dimensions
    if resize_shape is not None:
        mask1 = resize(mask1, resize_shape, preserve_range=True, anti_aliasing=False)
        mask2 = resize(mask2, resize_shape, preserve_range=True, anti_aliasing=False)

    # Ensure masks are binary
    mask1_binary = mask1 > 0
    mask2_binary = mask2 > 0

    # Convert binary masks to sets of points [(x1, y1), (x2, y2), ...]
    points1 = np.argwhere(mask1_binary)
    points2 = np.argwhere(mask2_binary)

    # Calculate directed Hausdorff distances and return the max (Hausdorff distance)
    hd1 = directed_hausdorff(points1, points2)[0]
    hd2 = directed_hausdorff(points2, points1)[0]

    return max(hd1, hd2)


def average_surface_distance(mask1, mask2, pixel_spacing=1.0):
    mask1_flat = mask1.squeeze()
    mask2_flat = mask2.squeeze()
    dist1 = distance_transform_edt(mask1_flat, sampling=pixel_spacing)
    dist2 = distance_transform_edt(mask2_flat, sampling=pixel_spacing)
    asd_1_to_2 = np.mean(dist1[mask2_flat > 0])
    asd_2_to_1 = np.mean(dist2[mask1_flat > 0])
    return (asd_1_to_2 + asd_2_to_1) / 2


def evaluate_ensemble(ensemble_predictions, ground_truth_masks, resize_shape=(256, 256)):
    dsc_scores, rvd_scores, hd_scores, asd_scores = [], [], [], []
    total = len(ensemble_predictions)
    for i, (pred_mask, gt_mask) in enumerate(zip(ensemble_predictions, ground_truth_masks)):
        print(f"Evaluating {i+1}/{total}")

    for pred_mask, gt_mask in zip(ensemble_predictions, ground_truth_masks):
        # Resize masks if necessary
        pred_mask_resized = resize(pred_mask, resize_shape, preserve_range=True, anti_aliasing=True)
        gt_mask_resized = resize(gt_mask, resize_shape, preserve_range=True, anti_aliasing=True)

        # Convert to binary if not already
        pred_mask_binary = pred_mask_resized > 0.5  # Adjust threshold as needed
        gt_mask_binary = gt_mask_resized > 0.5

        # Calculate metrics
        dsc = dice_similarity_coefficient(pred_mask_binary, gt_mask_binary)
        rvd = relative_volume_difference(pred_mask_binary, gt_mask_binary)
        hd = hausdorff_distance(pred_mask_binary, gt_mask_binary)  # Assumes masks are compatible for Hausdorff distance
        asd = average_surface_distance(pred_mask_binary, gt_mask_binary)  # Adjust for your specific implementation

        # Append scores
        dsc_scores.append(dsc)
        rvd_scores.append(rvd)
        hd_scores.append(hd)
        asd_scores.append(asd)

    # Calculate mean scores
    mean_dsc = np.mean(dsc_scores)
    mean_rvd = np.mean(rvd_scores)
    mean_hd = np.mean(hd_scores)
    mean_asd = np.mean(asd_scores)

    return mean_dsc, mean_rvd, mean_hd, mean_asd



def majority_vote_vectorized(model_predictions, num_classes):
    # Assuming model_predictions shape is (M, N, H, W)
    # Flatten the predictions to shape (M, N*H*W)
    flattened_predictions = model_predictions.reshape(model_predictions.shape[0], -1)

    # Initialize the result array
    ensemble_flat = np.zeros(flattened_predictions.shape[1], dtype=np.int32)

    # Vectorized voting for each pixel
    for pixel_index in range(flattened_predictions.shape[1]):
        # Extract votes for the current pixel across all models
        pixel_votes = flattened_predictions[:, pixel_index]
        # Perform majority vote and store the result
        ensemble_flat[pixel_index] = np.bincount(pixel_votes, minlength=num_classes).argmax()

    # Reshape the flat ensemble array back to the original image shape (N, H, W)
    ensemble_predictions = ensemble_flat.reshape(model_predictions.shape[1:])

    return ensemble_predictions


def load_predictions(prediction_dirs, image_filenames, output_shape):
    all_model_predictions = []
    for dir_path in prediction_dirs:
        model_predictions = []
        dir_path = Path(dir_path)
        for filename in image_filenames:
            prefixed_filename = "predicted_mask_" + filename
            file_path = dir_path / prefixed_filename
            print(f"Loading file: {file_path}")
            try:
                prediction = imread(str(file_path))
                # Resize prediction to the desired output shape
                prediction_resized = resize(prediction, output_shape, preserve_range=True, anti_aliasing=True).astype(prediction.dtype)
                model_predictions.append(prediction_resized)
            except FileNotFoundError:
                print(f"File not found: {file_path}")
                return None
        all_model_predictions.append(np.stack(model_predictions))
    return np.array(all_model_predictions)

def visualize_difference(prediction, ground_truth):
    difference = np.abs(prediction - ground_truth)

    plt.figure(figsize=(10, 10))
    plt.imshow(difference, cmap='hot')
    plt.title('Difference Map')
    plt.colorbar()
    plt.axis('off')
    plt.show()




def main():
    base_dir = "test/mask"
    models = ["model1", "model2", "model3"]  # Update this list with your actual models
    prediction_dirs = [os.path.join(base_dir, model) for model in models]

    image_dir = "test/image"
    image_filenames = os.listdir(image_dir)  # Assumes filenames match between images and predictions

    # Define the common output shape for all prediction masks
    # This should be a tuple (height, width), e.g., (256, 256)
    output_shape = (256, 256)  # Adjust this based on your requirements

    # Load model predictions, now including the output_shape argument
    model_predictions = load_predictions(prediction_dirs, image_filenames, output_shape)

    num_classes = 2  # Adjust based on your segmentation task

    # Apply majority voting to model predictions
    ensemble_predictions = majority_vote_vectorized(model_predictions, num_classes)

    # Ensemble_predictions contains the final segmentation results
    # Add any post-processing, evaluation, or saving operations here

    # Load ground truth masks for validation set
    ground_truth_dir = Path('test/mask/ground_truth')
    ground_truth_masks = load_ground_truth_masks(ground_truth_dir, image_filenames)

    # Assuming the loading of ensemble_predictions and ground_truth_masks is done before this
    resize_shape = (256, 256)  # Example, adjust as necessary
    mean_dsc, mean_rvd, mean_hd, mean_asd = evaluate_ensemble(ensemble_predictions, ground_truth_masks, resize_shape)

    print(f"Mean Dice Similarity Coefficient: {mean_dsc}")
    print(f"Mean Relative Volume Difference: {mean_rvd}")
    print(f"Mean Hausdorff Distance: {mean_hd}")
    print(f"Mean Average Surface Distance: {mean_asd}")

    # Visualize the difference for each image in your dataset
    for i in range(len(ensemble_predictions)):
        print(f"Visualizing difference for image: {image_filenames[i]}")
        visualize_difference(ensemble_predictions[i], ground_truth_masks[i])


if __name__ == "__main__":
    main()
