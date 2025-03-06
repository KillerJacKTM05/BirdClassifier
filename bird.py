import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# paths & params
train_dir = "/Train"
test_dir = "/Test"
model_path = "bird_classifier_model.h5"
img_size = (224, 224)
batch_size = 32

# Data Preprocessing & Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

# CNN Model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(224,224,3)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64, (3,3), activation="relu"),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(128, (3,3), activation="relu"),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(train_generator.num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_generator, validation_data=test_generator, epochs=20)
model.save(model_path)

# print(f"Model saved at {model_path}")

# Load Model
def predict_image(image_path, model_path):
    model = keras.models.load_model(model_path)
    img = keras.preprocessing.image.load_img(image_path, target_size=img_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0  # Normalize

    predictions = model.predict(img_array)
    class_index = predictions.argmax()
    class_labels = list(train_generator.class_indices.keys())

    return class_labels[class_index]

# Evaluate Model
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

# Classification Report
print("\nClassification Report:\n")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

# Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Display Sample Predictions
def display_sample_predictions(test_generator, model):
    x_test, y_test = next(test_generator)
    predictions = model.predict(x_test)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y_test, axis=1)
    class_labels = list(test_generator.class_indices.keys())

    plt.figure(figsize=(12, 6))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(x_test[i])
        plt.title(f"Pred: {class_labels[predicted_labels[i]]}\nTrue: {class_labels[true_labels[i]]}")
        plt.axis("off")
    plt.show()

display_sample_predictions(test_generator, model)

# Usage:
# prediction = predict_image("path_to_test_image.jpg", model_path)
# print("Predicted Bird:", prediction)
