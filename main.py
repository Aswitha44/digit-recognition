# Imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import ResNet50
from scipy.stats import mode
import seaborn as sns
import gzip

# Function to load KMNIST dataset
def load_kmnist(images_path, labels_path):
    with gzip.open(images_path, 'rb') as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    with gzip.open(labels_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    return images, labels

# Paths for KMNIST dataset
!wget http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz
!wget http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz
!wget http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz
!wget http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz

train_images_path = "train-images-idx3-ubyte.gz"
train_labels_path = "train-labels-idx1-ubyte.gz"
test_images_path = "t10k-images-idx3-ubyte.gz"
test_labels_path = "t10k-labels-idx1-ubyte.gz"

# Load MNIST
(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = tf.keras.datasets.mnist.load_data()

# Normalize Data
x_train_mnist = x_train_mnist[:10000] / 255.0  # Use 10,000 samples
y_train_mnist = y_train_mnist[:10000]
x_test_mnist = x_test_mnist[:2000] / 255.0     # Use 2,000 samples
y_test_mnist = y_test_mnist[:2000]

# Reshape for deep learning models
x_train_mnist_reshaped = x_train_mnist.reshape(-1, 28, 28, 1)
x_test_mnist_reshaped = x_test_mnist.reshape(-1, 28, 28, 1)

# One-hot encode labels
y_train_mnist_onehot = to_categorical(y_train_mnist, 10)
y_test_mnist_onehot = to_categorical(y_test_mnist, 10)

# Flatten MNIST for SVM
x_train_mnist_flatten = x_train_mnist.reshape(len(x_train_mnist), -1)
x_test_mnist_flatten = x_test_mnist.reshape(len(x_test_mnist), -1)

# SVM Model
def train_svm():
    svm = SVC(kernel='rbf', C=10, gamma=0.1, probability=True)
    svm.fit(x_train_mnist_flatten, y_train_mnist)
    y_pred_svm = svm.predict(x_test_mnist_flatten)
    accuracy = accuracy_score(y_test_mnist, y_pred_svm)
    print(f"SVM Accuracy on MNIST: {accuracy}")
    return svm, y_pred_svm

# YOLO-inspired CNN
def train_yolo():
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train_mnist_reshaped, y_train_mnist_onehot, epochs=5, batch_size=32, validation_split=0.1)
    accuracy = model.evaluate(x_test_mnist_reshaped, y_test_mnist_onehot, verbose=0)[1]
    print(f"YOLO Accuracy on MNIST: {accuracy}")
    return model

# ResNet Model
def train_resnet():
    x_train_resized = tf.image.resize(x_train_mnist_reshaped, (32, 32)).numpy()
    x_test_resized = tf.image.resize(x_test_mnist_reshaped, (32, 32)).numpy()

    x_train_rgb = np.repeat(x_train_resized, 3, axis=-1)
    x_test_rgb = np.repeat(x_test_resized, 3, axis=-1)

    base_model = ResNet50(weights=None, include_top=False, input_shape=(32, 32, 3))
    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train_rgb, y_train_mnist_onehot, epochs=5, batch_size=32, validation_split=0.1)
    accuracy = model.evaluate(x_test_rgb, y_test_mnist_onehot, verbose=0)[1]
    print(f"ResNet Accuracy on MNIST: {accuracy}")
    return model

# Train Models
svm, y_pred_svm = train_svm()
yolo_model = train_yolo()
resnet_model = train_resnet()

# Ensemble Learning
x_test_resized = tf.image.resize(x_test_mnist_reshaped, (32, 32)).numpy()
x_test_rgb = np.repeat(x_test_resized, 3, axis=-1)

resnet_preds = np.argmax(resnet_model.predict(x_test_rgb), axis=1)
yolo_preds = np.argmax(yolo_model.predict(x_test_mnist_reshaped), axis=1)

y_pred_svm = np.array(y_pred_svm)
final_preds = mode(np.vstack([y_pred_svm, resnet_preds, yolo_preds]), axis=0)[0].flatten()

ensemble_accuracy = accuracy_score(y_test_mnist, final_preds)
print(f"Ensemble Accuracy on MNIST: {ensemble_accuracy}")

# Find the best model
model_accuracies = {
    'SVM': accuracy_score(y_test_mnist, y_pred_svm),
    'YOLO': yolo_model.evaluate(x_test_mnist_reshaped, y_test_mnist_onehot, verbose=0)[1],
    'ResNet': resnet_model.evaluate(x_test_rgb, y_test_mnist_onehot, verbose=0)[1],
    'Ensemble': ensemble_accuracy
}
best_model_name = max(model_accuracies, key=model_accuracies.get)
print(f"Best Model: {best_model_name} with accuracy {model_accuracies[best_model_name]:.4f}")

# Predictions and Probabilities for Confusion Matrix and ROC Curves
def get_predictions_and_probabilities(model_name, model, x_test, x_test_flat=None):
    if model_name == 'SVM':
        y_pred = model.predict(x_test_flat)
        y_proba = model.predict_proba(x_test_flat)
    else:
        y_proba = model.predict(x_test)
        y_pred = np.argmax(y_proba, axis=1)
    return y_pred, y_proba

svm_preds, svm_probas = get_predictions_and_probabilities('SVM', svm, x_test_mnist_reshaped, x_test_flat=x_test_mnist_flatten)
yolo_preds, yolo_probas = get_predictions_and_probabilities('YOLO', yolo_model, x_test_mnist_reshaped)
resnet_preds, resnet_probas = get_predictions_and_probabilities('ResNet', resnet_model, x_test_rgb)

# Plot Confusion Matrices
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(10), yticklabels=np.arange(10))
    plt.title(f"Confusion Matrix: {title}")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

plot_confusion_matrix(y_test_mnist, svm_preds, "SVM")
plot_confusion_matrix(y_test_mnist, yolo_preds, "YOLO")
plot_confusion_matrix(y_test_mnist, resnet_preds, "ResNet")
plot_confusion_matrix(y_test_mnist, final_preds, "Ensemble")

# Plot ROC Curves
def plot_roc_curves(models, x_test, y_test, y_test_onehot):
    plt.figure(figsize=(10, 8))
    for model_name, (preds, probas) in models.items():
        fpr, tpr, _ = roc_curve(y_test_onehot.ravel(), probas.ravel())
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title("ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.show()

models_for_roc = {
    'SVM': (svm_preds, svm_probas),
    'YOLO': (yolo_preds, yolo_probas),
    'ResNet': (resnet_preds, resnet_probas),
    'Ensemble': (final_preds, np.eye(10)[final_preds])
}
plot_roc_curves(models_for_roc, x_test_mnist_reshaped, y_test_mnist, y_test_mnist_onehot)

# Display Predictions with Best Model
if best_model_name == 'SVM':
    best_predictions = svm_preds
elif best_model_name == 'YOLO':
    best_predictions = yolo_preds
elif best_model_name == 'ResNet':
    best_predictions = resnet_preds
else:
    best_predictions = final_preds

num_images_to_display = 10
plt.figure(figsize=(12, 6))
for i in range(num_images_to_display):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test_mnist[i], cmap='gray')
    plt.title(f"Pred: {best_predictions[i]}\nTrue: {y_test_mnist[i]}")
    plt.axis('off')

plt.suptitle(f"Recognized Images by Best Model: {best_model_name}", fontsize=16)
plt.tight_layout()
plt.show()

# Display Model Accuracies
plt.figure(figsize=(10, 6))
plt.bar(model_accuracies.keys(), model_accuracies.values(), color=['blue', 'green', 'red', 'purple'])
plt.title("Model Accuracies")
plt.ylabel("Accuracy")
plt.xlabel("Models")
plt.ylim(0, 1)
plt.show()