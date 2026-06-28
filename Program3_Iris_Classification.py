import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load Iris Dataset
iris = load_iris()

X = iris.data
y = iris.target

# Standardize Data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

learning_rates = [0.001, 0.01]
hidden_units = [8, 16]
dropouts = [0.0, 0.2]

trial = 1

for lr in learning_rates:
    for hidden in hidden_units:
        for dropout in dropouts:

            model = tf.keras.Sequential([
                tf.keras.layers.Dense(hidden, activation='relu'),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(3, activation='softmax')
            ])

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            history = model.fit(
                X_train,
                y_train,
                epochs=50,
                verbose=0,
                validation_data=(X_test, y_test)
            )

            train_loss = history.history['loss'][-1]
            val_acc = history.history['val_accuracy'][-1]

            print("\nTrial", trial)
            print("Learning Rate:", lr)
            print("Hidden Units:", hidden)
            print("Dropout:", dropout)
            print("Training Loss:", train_loss)
            print("Validation Accuracy:", val_acc)

            trial += 1

pred = model.predict(X_test, verbose=0)
pred = np.argmax(pred, axis=1)

accuracy = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred, average='weighted')
recall = recall_score(y_test, pred, average='weighted')
f1 = f1_score(y_test, pred, average='weighted')

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)
