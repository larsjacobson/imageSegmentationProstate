from pymongo import MongoClient
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import pydicom
from pydicom.filebase import DicomBytesIO
import numpy as np
from PIL import Image
from matplotlib.patches import Polygon

# MongoDB connection details
client = MongoClient('mongodb+srv://larsjacobson:4Mf%239x8E@eojgroup.6ifxr.mongodb.net/?retryWrites=true&w=majority')
db = client.prostateSegmentation_MRI
collection = db.input

# Directories to save masks, original images, and combined images
mask_dir = '/Users/lars.jacobson/PycharmProjects/imageSegmentation_MRI/train/mask'
image_dir = '/Users/lars.jacobson/PycharmProjects/imageSegmentation_MRI/train/image'
combined_dir = '/Users/lars.jacobson/PycharmProjects/imageSegmentation_MRI/train/combined'
os.makedirs(mask_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)
os.makedirs(combined_dir, exist_ok=True)

sas_token = "?sv=2022-11-02&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2024-01-18T21:34:25Z&st=2024-01-18T13:34:25Z&spr=https&sig=YMaXcmweC0NRsCM48VHEFAJE2ykuu6l2egUZhlqMGTU%3D"  # Replace with your SAS token

import numpy as np

def dicom_to_pil(dicom_data):
    image_array = dicom_data.pixel_array
    print(f"Processing DICOM image with shape: {image_array.shape} and data type: {image_array.dtype}")

    if image_array.ndim == 2:
        # Normalize the 16-bit image data to 0-255
        normalized_image = (image_array - image_array.min()) / (image_array.max() - image_array.min())
        image_array_8bit = (normalized_image * 255).astype('uint8')
        return np.array(Image.fromarray(image_array_8bit))
    else:
        print(f"Unsupported image shape: {image_array.shape}")
        return None

def get_authenticated_url(base_url):
    return base_url + sas_token

# Function to draw annotations (modified to use authenticated URL)
from urllib.parse import urlparse, unquote

def draw_annotations(image_url, annotations, patient_id):
    print(f"Processing image: {image_url}")
    authenticated_url = get_authenticated_url(image_url)
    response = requests.get(authenticated_url)

    if response.status_code == 200:
        try:
            dicom_data = pydicom.dcmread(DicomBytesIO(response.content), force=True)
            img_array = dicom_to_pil(dicom_data)
            if img_array is None:
                print(f"Skipping image: Unsupported format or failed conversion for URL {image_url}")
                return
        except Exception as e:
            print(f"Error processing DICOM file: {e}")
            return

        # Convert the PIL Image to a NumPy array for Matplotlib
        img = np.array(img_array)
        img_height, img_width = img.shape[:2]

        # Define a color mapping for different categories
        category_colors = {
            1: 'green',  # Example category_id and its corresponding color
            2: 'blue',
            3: 'red',
            # Add more category IDs and their colors as needed
        }

        # Convert NumPy array to PIL Image for saving
        img_pil = Image.fromarray(img_array)

        # Save the original image
        image_filename = f"{patient_id}_{os.path.basename(image_url).split('.')[0]}_image.png"
        image_path = os.path.join(image_dir, image_filename)
        img_pil.save(image_path)
        print(f"Saved original image to {image_path}")

        # Prepare to draw the mask (annotations only)
        fig_mask, ax_mask = plt.subplots()
        ax_mask.set_xlim([0, img_width])
        ax_mask.set_ylim([img_height, 0])
        ax_mask.axis('off')
        ax_mask.set_facecolor((0, 0, 0))  # Set the axes background to black

        for ann in annotations:
            category_id = ann['category_id']
            color = category_colors.get(category_id, 'white')  # This should remain as the annotation color

            for polygon in ann['segmentation']:
                scaled_polygon = [(x * img_width if i % 2 == 0 else x * img_height) for i, x in enumerate(polygon)]
                poly_patch = Polygon(np.array(scaled_polygon).reshape(-1, 2), closed=True, linewidth=2, edgecolor=color,
                                     facecolor='none')  # Annotations filled with 'color'
                ax_mask.add_patch(poly_patch)

        # Save the mask image
        fig_mask.patch.set_facecolor((0, 0, 0, 0))  # Set the figure background to black
        mask_filename = f"{patient_id}_{os.path.basename(image_url).split('.')[0]}_mask.png"
        mask_path = os.path.join(mask_dir, mask_filename)
        fig_mask.savefig(mask_path, dpi=300, bbox_inches='tight', pad_inches=0, facecolor=fig_mask.get_facecolor())
        plt.close(fig_mask)
        print(f"Saved mask to {mask_path}")

    # Save the combined image (original + annotations)
    fig_combined, ax_combined = plt.subplots()
    ax_combined.imshow(img, cmap='gray')

    for ann in annotations:
        category_id = ann['category_id']
        color = category_colors.get(category_id, 'white')  # Default to white if category not found

        for polygon in ann['segmentation']:
            scaled_polygon = [(x * img_width if i % 2 == 0 else x * img_height) for i, x in enumerate(polygon)]
            poly_patch = Polygon(np.array(scaled_polygon).reshape(-1, 2), closed=True, linewidth=2, edgecolor=color, facecolor=color, alpha=0.1)
            ax_combined.add_patch(poly_patch)

            combined_filename = f"{patient_id}_{os.path.basename(image_url).split('.')[0]}_combined.png"
            combined_path = os.path.join(combined_dir, combined_filename)
            fig_combined.savefig(combined_path, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close(fig_combined)
        else:
            print(f"Failed to fetch the image: HTTP {response.status_code}")



# Process each COCO file in the collection
for coco_file in collection.find():
    patient_id = coco_file['_patient']  # Extract the _patient field from the document
    for image in coco_file['images']:
        image_url = image['absolute_url']
        image_annotations = [ann for ann in coco_file['annotations'] if ann['image_id'] == image['id']]
        draw_annotations(image_url, image_annotations, patient_id)  # Pass the patient_id here