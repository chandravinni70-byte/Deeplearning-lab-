import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

# Load MNIST Dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize Data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build Model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train Model
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2
)

# Evaluate Model
loss, accuracy = model.evaluate(x_test, y_test)

print("Test Accuracy:", accuracy)

# Predict
pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(8, 8))
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Performance Metrics
accuracy = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred, average="weighted")
recall = recall_score(y_test, pred, average="weighted")
f1 = f1_score(y_test, pred, average="weighted")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

# Misclassified Images
predictions = model.predict(x_test, verbose=0)
predicted_labels = np.argmax(predictions, axis=1)

misclassified = np.where(predicted_labels != y_test)[0]

print("Total Misclassified Images:", len(misclassified))

for i in range(3):
    idx = misclassified[i]

    plt.figure(figsize=(2, 2))
    plt.imshow(x_test[idx], cmap="gray")
    plt.title(
        f"Image ID: {idx}\nTrue: {y_test[idx]}  Predicted: {predicted_labels[idx]}")
    plt.axis("off")
    plt.show()

    print("Image ID:", idx)
    print("True Label:", y_test[idx])
    print("Predicted Label:", predicted_labels[idx])
    print("-" * 40)
