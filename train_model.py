import os
import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ─────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────
IMAGE_FOLDER = r"D:\superior\streamlit model\archive\Brain Tumor\Brain Tumor"
CSV_PATH     = r"D:\superior\streamlit model\archive\Brain Tumor.csv"
IMG_SIZE     = 128
BATCH_SIZE   = 32
EPOCHS       = 20
MODEL_SAVE   = r"D:\superior\streamlit model\brain_tumor_model.h5"

# ─────────────────────────────────────────
# 2. LOAD CSV LABELS
# ─────────────────────────────────────────
print("Loading CSV...")
df = pd.read_csv(CSV_PATH)
print(f"  CSV columns: {df.columns.tolist()}")
print(df.head())

# The label column is named 'Class' (1=Tumor, 0=No Tumor)
# Image column is 'Image'
# Adjust if your columns differ after seeing the print above
LABEL_COL = "Class"
IMAGE_COL  = "Image"

# ─────────────────────────────────────────
# 3. LOAD IMAGES USING CSV LABELS
# ─────────────────────────────────────────
print("\nLoading images...")
X, y = [], []

for _, row in df.iterrows():
    img_name = str(row[IMAGE_COL]).strip()
    label    = int(row[LABEL_COL])

    # Try with and without extension
    for fname in [img_name, img_name + ".jpg"]:
        img_path = os.path.join(IMAGE_FOLDER, fname)
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                X.append(img)
                y.append(label)
            break

print(f"  Loaded: {len(X)} images | Tumor: {sum(1 for l in y if l==1)} | No Tumor: {sum(1 for l in y if l==0)}")

if len(X) == 0:
    print("\n❌ No images loaded. Check IMAGE_COL name matches your CSV.")
    exit()

X = np.array(X, dtype="float32") / 255.0
y = np.array(y)

# ─────────────────────────────────────────
# 4. TRAIN / TEST SPLIT
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

# ─────────────────────────────────────────
# 5. DATA AUGMENTATION
# ─────────────────────────────────────────
datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(X_train)

# ─────────────────────────────────────────
# 6. BUILD CNN MODEL
# ─────────────────────────────────────────
model = Sequential([
    Conv2D(32, (3,3), activation="relu", padding="same", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(),
    Conv2D(32, (3,3), activation="relu", padding="same"),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Conv2D(64, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    Conv2D(64, (3,3), activation="relu", padding="same"),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Conv2D(128, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    Conv2D(128, (3,3), activation="relu", padding="same"),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Flatten(),
    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# ─────────────────────────────────────────
# 7. TRAIN
# ─────────────────────────────────────────
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
    ModelCheckpoint(MODEL_SAVE, save_best_only=True, monitor="val_accuracy")
]

print("\nTraining...")
model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    callbacks=callbacks
)

# ─────────────────────────────────────────
# 8. EVALUATE
# ─────────────────────────────────────────
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {acc*100:.2f}%")
print(f"✅ Model saved to: {MODEL_SAVE}")