import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# XOR Dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

y = np.array([0, 1, 1, 0])

experiments = [
    ("sigmoid", 0.01),
    ("sigmoid", 0.1),
    ("relu", 0.01),
    ("relu", 0.1)
]

results = []

for activation, lr in experiments:

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(4, input_shape=(2,), activation=activation),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        X,
        y,
        epochs=100,
        verbose=0
    )

    loss = history.history['loss'][-1]
    acc = history.history['accuracy'][-1]

    results.append([activation, lr, loss, acc])

print("\nResults")
for r in results:
    print(r)

pred = model.predict(X, verbose=0)
pred = (pred > 0.5).astype(int)

accuracy = accuracy_score(y, pred)
precision = precision_score(y, pred)
recall = recall_score(y, pred)
f1 = f1_score(y, pred)

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)
