import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ProgbarLogger, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from skimage.io import imread
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import tensorflow.keras.backend as k

# Data Loading and Preprocessing
from skimage.color import gray2rgb

def load_png_image(path):
    """Load a PNG image, normalize it, and ensure it is in RGB format."""
    print(f"Loading image: {path}")
    image = imread(path)

    # If grayscale, convert to RGB
    if image.ndim == 2:
        image = gray2rgb(image)
    elif image.shape[2] == 4:  # If RGBA, drop the alpha channel
        image = image[..., :3]

    image = image.astype(np.float32)
    image /= np.max(image)  # Normalize to [0, 1]
    return image


def calculate_class_weights(masks):
    """Calculate class weights based on the distribution of mask pixels."""
    # Flatten the masks to get all labels in one array
    labels_flat = masks.argmax(axis=-1).flatten()
    # Determine unique class labels
    classes = np.unique(labels_flat)
    # Calculate class weights for these classes
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels_flat)
    # Create a dictionary mapping class indices to their respective weights
    class_weights_dict = {class_id: weight for class_id, weight in zip(classes, class_weights)}

    return class_weights_dict

def load_mask_image(path):
    """Load a color mask and convert it to one-hot encoded classes."""
    mask = imread(path)
    mask_class_labels = convert_colors_to_labels(mask, tolerance=15)
    mask_one_hot = to_categorical(mask_class_labels, num_classes=4)  # Specify 4 classes here
    return mask_one_hot




def convert_colors_to_labels(mask, tolerance=15):
    """Convert mask colors to class labels with a tolerance for color matching."""
    # Define colors as arrays
    colors = {
        0: np.array([0, 0, 0]),    # Background
        1: np.array([0, 128, 0]),  # Green
        2: np.array([0, 0, 255]),  # Blue
        3: np.array([255, 0, 0])   # Red
    }
    mask_class_labels = np.zeros(mask.shape[:2], dtype=np.int32)
    for class_label, color in colors.items():
        # Adjust calculation to ignore the alpha channel if present in the mask
        mask_rgb = mask[..., :3] if mask.shape[-1] == 4 else mask
        distance = np.sqrt(np.sum((mask_rgb - color) ** 2, axis=-1))
        mask_class_labels[distance < tolerance] = class_label

    # Debugging: Print unique class labels and their counts
    unique, counts = np.unique(mask_class_labels, return_counts=True)
    print("Class labels and counts:", dict(zip(unique, counts)))
    return mask_class_labels



def load_data(image_dir, mask_dir):
    """Load and preprocess image and mask data."""
    images, masks = [], []
    for filename in os.listdir(image_dir):
        if filename.endswith('_image.png'):
            img_path = os.path.join(image_dir, filename)
            mask_path = os.path.join(mask_dir, filename.replace('_image.png', '_mask.png'))
            img = load_png_image(img_path)
            mask = load_mask_image(mask_path)
            # Resize images and masks separately
            resized_img = resize(img, (128, 128), anti_aliasing=True)
            resized_mask = resize(mask, (128, 128), anti_aliasing=True, preserve_range=True)
            images.append(resized_img)
            masks.append(resized_mask)
    return np.array(images), np.array(masks)



# U-Net Model
def unet(input_shape=(128, 128, 3)):
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

    last_layer = conv9
    outputs = Conv2D(4, (1, 1), activation="softmax")(last_layer)

    model = Model(inputs=inputs, outputs=outputs, name='unet')
    return model


# Main Execution
if __name__ == "__main__":
    print("Script started.")

    # Data Preparation
    image_dir, mask_dir = 'train/image', 'train/mask'
    images, masks = load_data(image_dir, mask_dir)

    # Calculate class weights based on the training masks
    class_weights = calculate_class_weights(masks)

    # Splitting data
    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.1, random_state=42)

    # Data Augmentation
    data_gen_args = dict(rotation_range=90,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # Prepare the generators for training
    seed = 1
    image_generator = image_datagen.flow(X_train, batch_size=32, seed=seed)
    mask_generator = mask_datagen.flow(y_train, batch_size=32, seed=seed)

    # Combine generators into one which yields image and masks
    train_generator = zip(image_generator, mask_generator)

    # Model Compilation
    model = unet((128, 128, 3))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'categorical_crossentropy'])

    # Callbacks
    progbar_logger = ProgbarLogger(count_mode='steps', stateful_metrics=None)
    model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)

    # Model Training with class weights
    history = model.fit(
        train_generator,
        steps_per_epoch=len(X_train) // 32,
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=[progbar_logger, model_checkpoint, reduce_lr],
        class_weight=class_weights  # Apply the calculated class weights here
    )

    # Save the final model
    model.save('prostate_segmentation_model_multiple_classes_final')
    print("Final model saved. Script finished.")

    print("Training image shape:", X_train.shape)
    print("Training mask shape:", y_train.shape)

