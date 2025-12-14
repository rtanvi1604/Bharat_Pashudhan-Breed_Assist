# ==========================
# 1. Import Libraries
# ==========================
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

# ==========================
# 2. Dataset Path
# ==========================
dataset_dir ="C:/Users/lohit/OneDrive/Desktop/IMG_CLASSIFICATION/Cattle Breeds"  # <-- change path if needed

# ==========================
# 3. Data Preprocessing
# ==========================
img_size = 224
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 80% train, 20% validation
)

train_gen = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_gen = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# ==========================
# 4. Load Pretrained Model (ResNet50)
# ==========================
base_model = ResNet50(weights='imagenet', include_top=False,
                      input_shape=(img_size, img_size, 3))

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Custom top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# ==========================
# 5. Compile Model
# ==========================
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ==========================
# 6. Train Model
# ==========================
history = model.fit(
    train_gen,
    epochs=10,   # increase to 30+ for better accuracy
    validation_data=val_gen
)

# ==========================
# 7. Evaluate Model
# ==========================
loss, acc = model.evaluate(val_gen, verbose=0)
print(f"Validation Accuracy: {acc*100:.2f}%")

# ==========================
# 8. Plot Training Curves
# ==========================
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
