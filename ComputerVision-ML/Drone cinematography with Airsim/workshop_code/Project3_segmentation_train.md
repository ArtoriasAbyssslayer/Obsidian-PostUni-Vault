# Image segmentation with a U-Net-like architecture


**Author:** [fchollet](https://twitter.com/fchollet), Sotirios Papadopoulos<br>
**Date created:** 2022/08/18<br>
**Description:** Image segmentation model trained from scratch on the generated AirSim dataset (Code modified from https://keras.io/examples/vision/oxford_pets_image_segmentation/).

## Prepare paths of input images and target segmentation masks

<p>It is expected that the RGB images will be stored inside the <i>segmentation_dataset/input_imgs</i> folder, and the segmentation ground truth maps inside <i>segmentation_dataset/target_imgs</i>.</p>
The batch_size hyperparameter is directly linked to the size of GPU memory needed for the training. In case of a "<b>RuntimeError: CUDA error: out of memory</b>" error, you may reduce the batch size according to your hardware needs. A larger batch size may lead to better accuracy.


```python
import os

input_dir = "segmentation_dataset/input_imgs/"
target_dir = "segmentation_dataset/target_imgs/"
img_size = (144, 256)
num_classes = 6
batch_size = 4

input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".png")
    ]
)
target_img_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".png") and not fname.startswith(".")
    ]
)

print("Number of samples:", len(input_img_paths))

for input_path, target_path in zip(input_img_paths[:10], target_img_paths[:10]):
    print(input_path, "|", target_path)
```

## What does one input image and corresponding segmentation mask look like?

Running this cell will display the image n.8 with its corresponding ground truth. To view a different image change the img_to_show parameter. 


```python
from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
from PIL import ImageOps
import PIL
import numpy as np

# Display input image #8
img_to_show = 8
display(Image(filename=input_img_paths[img_to_show]))

# Display auto-contrast version of corresponding target (per-pixel categories)
img =  PIL.Image.fromarray(np.asarray(ImageOps.autocontrast(load_img(target_img_paths[img_to_show]))))

display(img)
```

## Prepare `Sequence` class to load & vectorize batches of data

This cell is responsible for turning a batch of image paths into a numpy array containing the corresponding images.


```python
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img


class AirsimDataset(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
        return x, y

```

## Prepare U-Net Xception-style model

This cell builds the semantic segmentation neural network using keras. The network, like most pixel-level prediction networks follows the Encoder-Decoder with skip connections architectures where there are two distinct parts, the Encoding part and the Decoding part. 

During the encoding part, the input image spatial dimensions are progressively reduced. In the decoding part, in contrast, the spatial dimensions of the activation maps are progressively returning to the original input dimensions.

Feel free to make any adjustments!


```python
from tensorflow.keras import layers


def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Build model
model = get_model(img_size, num_classes)
model.summary()
```

## Set aside a validation split

In this cell, we randomly split the dataset into data used for training and data used for validation. By default we keep 150 samples as the validation set. 


```python
import random

# Split our img paths into a training and a validation set
val_samples = 150
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)
train_input_img_paths = input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:-val_samples]
val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[-val_samples:]

# Instantiate data Sequences for each split
train_gen = AirsimDataset(
    batch_size, img_size, train_input_img_paths, train_target_img_paths
)
val_gen = AirsimDataset(batch_size, img_size, val_input_img_paths, val_target_img_paths)
```

## Train the model

In this cell we train the network using the rmsprop optimizer algorithm and the sparce categorical crossentropy loss function (typical for semantic segmentation). We train the network for 15 epochs (how many times the training procedure will traverse the whole training dataset).


```python
# Configure the model for training.
# We use the "sparse" version of categorical_crossentropy
# because our target data is integers.
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

callbacks = [
    #keras.callbacks.ModelCheckpoint("trained_segmentation model.h5", save_best_only=True)
]

# Train the model, doing validation at the end of each epoch.
epochs = 15
model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)
model.save('trained_segmentation model.h5')
```

## Visualize predictions


```python
# Generate predictions for all images in the validation set

val_gen = AirsimDataset(batch_size, img_size, val_input_img_paths, val_target_img_paths)
val_preds = model.predict(val_gen)
#model = keras.models.load_model('trained_segmentation model.h5')


def display_mask(i):
    """Quick utility to display a model's prediction."""
    mask = np.argmax(val_preds[i], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    img = ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
    display(img)


# Display results for validation image #3
i = 3

# Display input image
display(Image(filename=val_input_img_paths[i]))

# Display ground-truth target mask
img = ImageOps.autocontrast(load_img(val_target_img_paths[i]))
display(img)

# Display mask predicted by our model
display_mask(i)  # Note that the model only sees inputs at 150x150.
```


```python

```
