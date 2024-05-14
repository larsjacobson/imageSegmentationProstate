import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ProgbarLogger
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from skimage.io import imread
import os
import tensorflow.keras.backend as k

# -------------------------------
# Data Loading and Preprocessing
# -------------------------------

def load_png_image(path):
    """Load a PNG image and normalize it."""
    print(f"Loading image: {path}")
    image = imread(path)
    image = image.astype(np.float32)
    image /= np.max(image)  # Normalize to [0, 1]
    return image

def load_mask_image(path, num_classes=4):
    """Load a color mask and convert it to one-hot encoded classes."""
    mask = imread(path)
    mask_class_labels = convert_colors_to_labels(mask)
    mask_one_hot = to_categorical(mask_class_labels, num_classes=num_classes)
    return mask_one_hot

def convert_colors_to_labels(mask):
    """Convert mask colors to class labels."""
    mask_class_labels = np.zeros(mask.shape[:2], dtype=np.int32)
    # Map each color to a class index (assuming RGB format)
    mask_class_labels[np.all(mask == [0, 0, 0], axis=-1)] = 0  # Background
    mask_class_labels[np.all(mask == [0, 255, 0], axis=-1)] = 1  # Green
    mask_class_labels[np.all(mask == [0, 0, 255], axis=-1)] = 2  # Blue
    mask_class_labels[np.all(mask == [255, 0, 0], axis=-1)] = 3  # Red
    return mask_class_labels

def load_data(image_dir, mask_dir, num_classes=4):
    """Load and preprocess image and mask data."""
    images, masks = [], []
    for filename in os.listdir(image_dir):
        if filename.endswith('_image.png'):
            img_path = os.path.join(image_dir, filename)
            mask_path = os.path.join(mask_dir, filename.replace('_image.png', '_mask.png'))
            img, mask = load_png_image(img_path), load_mask_image(mask_path, num_classes)
            images.append(resize(img, (128, 128, 1), anti_aliasing=True))
            masks.append(resize(mask, (128, 128, num_classes), anti_aliasing=True))
    return np.array(images), np.array(masks)

# --------------
# U-Net Model
# --------------

def unet(input_shape, num_classes=4):
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

    last_layer = conv9

    outputs = tf.keras.layers.Conv2D(num_classes, 1, activation='softmax')(last_layer)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='unet')

    return model

# ----------------
# Main Execution
# ----------------

if __name__ == "__main__":
    print("Script started.")

    # Data Preparation
    image_dir, mask_dir = 'train/image', 'train/mask'
    images, masks = load_data(image_dir, mask_dir, num_classes=4)

    # Data Resizing and Splitting
    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.1, random_state=42)
    print(f"Data shapes - Images: {X_train.shape}, Masks: {y_train.shape}")

    # Model Compilation
    model = unet((128, 128, 1), num_classes=4)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Model Training
    progbar_logger = ProgbarLogger(count_mode='steps', stateful_metrics=None)
    history = model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_val, y_val), callbacks=[progbar_logger])

    # Save the Model
    model.save('prostate_segmentation_model_multiple_classes_old.h5')
    print("Model saved. Script finished.")