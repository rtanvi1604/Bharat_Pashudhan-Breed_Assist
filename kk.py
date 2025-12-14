# Import all necessary libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import json
import envs
# Disable some overly verbose logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define important parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 10
# REMOVED THE HARD-CODED NUM_CLASSES HERE

# Define the paths to your data
train_dir = "C:/Users/lohit/OneDrive/Desktop/IMG_CLASSIFICATION/Cattle Breeds"  
validation_dir = "C:/Users/lohit/OneDrive/Desktop/IMG_CLASSIFICATION/Cattle Breeds"

# Step 2: Data Preprocessing & Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Load images from directories
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

validation_data = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False)

# !!! CRITICAL FIX !!!
# Get the number of classes DIRECTLY from the training data generator
NUM_CLASSES = len(train_data.class_indices)
print(f"Number of classes detected: {NUM_CLASSES}")
# This will print the classes and their assigned indices
print(f"Class indices: {train_data.class_indices}")

# Step 3: Build the Model using Transfer Learning
base_model = keras.applications.MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

model = Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(128, activation='relu'),
    # The output layer now uses the dynamically found NUM_CLASSES
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Step 4: Compile the Model
# Since we have more than 2 classes, we MUST use 'categorical_crossentropy'
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Step 5: Train the Model
history = model.fit(
    train_data,
    steps_per_epoch = train_data.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_data,
    validation_steps = validation_data.samples // BATCH_SIZE
)

# ... (The rest of the code for plotting and saving remains the same)
# Step 6: Evaluate the Accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Step 7: Save the trained model
model.save('my_image_classifier.h5')
print("Model saved as 'my_image_classifier.h5'")
# ... (your existing training code)

# NEW CODE: Save the class indices to a JSON file

with open('class_indices.json', 'w') as f:
    json.dump(train_data.class_indices, f)
print("Class indices saved as 'class_indices.json'")