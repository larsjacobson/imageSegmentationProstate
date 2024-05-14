from skimage.io import imread
import os

def check_image_shape(image_path):
    # Load an image
    image = imread(image_path)

    # Print the shape of the image
    print(f"The shape of the image is: {image.shape}")

# Example usage
image_dir = 'train/image'
sample_image_path = os.path.join(image_dir, os.listdir(image_dir)[0])  # Path to the first image in the directory
check_image_shape(sample_image_path)