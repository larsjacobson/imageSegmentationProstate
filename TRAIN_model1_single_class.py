import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ProgbarLogger
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from skimage.io import imread
import os
import tensorflow.keras.backend as k

# Data Loading and Preprocessing Functions
def load_png_image(path):
    """Load a PNG image."""
    print(f"Loading image: {path}")
    image = imread(path)
    image = image.astype(np.float32)
    image /= np.max(image)  # Normalize to [0, 1]
    return image

def load_mask_image(path):
    """Load a mask image (assumed to be in PNG format)."""
    print(f"Loading mask: {path}")
    mask = imread(path, as_gray=True)
    return mask

def load_data(image_dir, mask_dir):
    """Load and preprocess image and mask data."""
    print("Loading data...")
    images = []
    masks = []

    for filename in os.listdir(image_dir):
        if filename.endswith('_image.png'):
            img_path = os.path.join(image_dir, filename)
            mask_path = os.path.join(mask_dir, filename.replace('_image.png', '_mask.png'))

            img = load_png_image(img_path)
            mask = load_mask_image(mask_path)

            img_resized = resize(img, (128, 128, 1), anti_aliasing=True)
            mask_resized = resize(mask, (128, 128, 1), anti_aliasing=True)

            images.append(img_resized)
            masks.append(mask_resized)

    return np.array(images), np.array(masks)

# U-Net Model Function
def unet(input_shape):
    """Build a U-Net model."""
    k.clear_session()
    inputs = Input(input_shape)
    conv1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(conv9)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='UNet')

    return model

# Main Script Execution
if __name__ == "__main__":
    print("Script started.")
    # Load data
    image_dir = 'train/image'
    mask_dir = 'train/mask'
    images, masks = load_data(image_dir, mask_dir)

    # Resize and split data
    print("Resizing and splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.1, random_state=42)
    print("Data ready for training.")

    # Build and compile the model
    model = unet((128, 128, 1))
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    print("Model compiled.")

    # Train the model
    print("Starting training...")
    history = model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_val, y_val), callbacks=[ProgbarLogger()])
    print("Training complete.")

    # Save the model
    model.save('prostate_segmentation_model_single_class.h5')
    print("Model saved. Script finished.")