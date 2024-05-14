from tensorflow.keras.models import load_model
from skimage.transform import resize
from skimage.io import imread, imsave
from skimage import img_as_ubyte
import numpy as np
import os
from skimage.transform import resize

# Load the trained model from the directory
model = load_model('prostate_segmentation_model_multiple_classes_final')


def prepare_image(img_path):
    """
    Load an image, normalize it, resize it to fit the model input, and ensure it has the correct shape.
    """
    img = imread(img_path)
    img = resize(img, (128, 128), anti_aliasing=True)
    img = img.astype(np.float32) / np.max(img)  # Normalize to [0, 1]

    # Ensure img has shape (128, 128, 3) before expanding the batch dimension
    if img.ndim == 2:  # If the image is grayscale, convert to RGB
        img = np.stack((img,) * 3, axis=-1)
    elif img.shape[-1] == 4:  # If the image has an alpha channel, drop it
        img = img[..., :3]

    img = np.expand_dims(img, axis=0)  # Add batch dimension to get shape (1, 128, 128, 3)
    return img


def predict_mask(model, img):
    """
    Use the model to predict a mask for the input image.
    """
    pred_mask = model.predict(img)
    pred_mask = np.argmax(pred_mask, axis=-1)  # Convert probabilities to class labels
    pred_mask = pred_mask[0]  # Remove batch dimension
    return pred_mask


def labels_to_colors(pred_mask):
    """
    Convert predicted class labels back to a color mask for visualization.
    """
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


# Directory for saving the combined images
combined_save_dir = 'test/combined'
if not os.path.exists(combined_save_dir):
    os.makedirs(combined_save_dir)

# Directory for saving the predicted masks
mask_save_dir = 'test/mask/model5'
if not os.path.exists(mask_save_dir):
    os.makedirs(mask_save_dir)

# Directory containing your test images
test_img_dir = 'test/image'

# List all image paths
test_img_paths = [os.path.join(test_img_dir, fname) for fname in os.listdir(test_img_dir) if fname.endswith('.png')]

# Loop through all test images
for img_path in test_img_paths:
    test_img = prepare_image(img_path)
    predicted_mask = predict_mask(model, test_img)
    color_mask = labels_to_colors(predicted_mask)

    # Load the original image and resize it to match the shape of the predicted mask
    original_img = imread(img_path)
    original_img = resize(original_img, color_mask.shape[:2], anti_aliasing=True)

    # Convert grayscale image to 3-channel if needed
    if original_img.ndim == 2:
        original_img = np.stack((original_img,) * 3, axis=-1)

    # Ensure both arrays have the same data type and value range
    original_img = original_img.astype(np.float32)
    color_mask = color_mask.astype(np.float32)

    # Ensure both arrays have the same data type and value range
    original_img = original_img.astype(np.float32)
    color_mask = color_mask.astype(np.float32)

    # Combine original image and predicted mask
    combined_img = 0.6 * original_img + 0.4 * color_mask  # Adjust blending as needed
    combined_img = np.clip(combined_img, -1, 1)  # Clip values to the range [-1, 1]

    # Convert the combined image to unsigned byte
    combined_img = img_as_ubyte(combined_img)

    # Save the combined image
    combined_save_path = os.path.join(combined_save_dir, os.path.basename(img_path))
    imsave(combined_save_path, combined_img)

    print(f"Saved combined image to: {combined_save_path}")

    # Extract the base filename from 'img_path' and prepend with 'predicted_mask_'
    filename_with_prefix = f"predicted_mask_{os.path.basename(img_path)}"

    # Update 'mask_save_path' to include the modified filename
    mask_save_path = os.path.join(mask_save_dir, filename_with_prefix)

    # Convert color mask to uint8
    color_mask_uint8 = (color_mask * 255).astype(np.uint8)

    # Save the color mask
    imsave(mask_save_path, color_mask_uint8)

    # Print the path where the predicted mask was saved
    print(f"Saved predicted mask to: {mask_save_path}")
