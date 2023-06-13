import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
import pathlib
import random

# Set the random seeds for reproducibility
random.seed(6)
np.random.seed(6)
tf.random.set_seed(6)

# Set dataset directory's
train_dir = 'Subset_of_posted_Items/train'
val_dir = 'Subset_of_posted_Items/validation'
test_dir = 'Subset_of_posted_Items/test'

train_data_dir = pathlib.Path(train_dir)
val_data_dir = pathlib.Path(val_dir)
test_data_dir = pathlib.Path(test_dir)

# Count the number of images in each directory
train_image_count = len(list(train_data_dir.glob('*/*.jpg')))
validation_image_count = len(list(val_data_dir.glob('*/*.jpg')))
test_image_count = len(list(test_data_dir.glob('*/*.jpg')))
print("Number of images in train directory:", train_image_count)
print("Number of images in validation directory:", validation_image_count)
print("Number of images in test directory:", test_image_count)
print("\n")


batch_size = 32
img_height = 180
img_width = 180

# Create the training/validation/testing datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Find and print all the class names
class_names = train_ds.class_names
print("Class names: ", class_names)
print("\n")

# Configure the datasets for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)

data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

# Create the models layers
model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()
print("\n")

# Training the model on the training data, and using validation data for validation
epochs = 25
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

# Calculate the average accuracy and loss across epochs
avg_train_accuracy = np.mean(acc)
avg_val_accuracy = np.mean(val_acc)
avg_train_loss = np.mean(loss)
avg_val_loss = np.mean(val_loss)

# Print the average accuracy and loss
print("Average Training Accuracy:", avg_train_accuracy)
print("Average Validation Accuracy:", avg_val_accuracy)
print("Average Training Loss:", avg_train_loss)
print("Average Validation Loss:", avg_val_loss)
print("\n")

# Plotting the training/validation accuracy and loss
epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_ds)
print("\n")
print("Evaluating on the test dataset...")
print("Average Test Accuracy:", test_accuracy)
print("Average Test Loss:", test_loss)
