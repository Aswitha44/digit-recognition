# Additional imports for noise generation
import cv2
from skimage.util import random_noise

# Function to add Gaussian noise
def add_gaussian_noise(images, mean=0, var=0.01):
    noisy_images = []
    for image in images:
        noise = np.random.normal(mean, var ** 0.5, image.shape)
        noisy_image = np.clip(image + noise, 0, 1)
        noisy_images.append(noisy_image)
    return np.array(noisy_images)

# Function to add Salt-and-Pepper noise
def add_salt_and_pepper_noise(images, amount=0.02):
    noisy_images = []
    for image in images:
        noisy_image = random_noise(image, mode='s&p', amount=amount)
        noisy_images.append(noisy_image)
    return np.array(noisy_images)

# Evaluate models on noisy images
def evaluate_on_noisy_images(models, x_test, y_test, noise_type='Gaussian', **kwargs):
    if noise_type == 'Gaussian':
        x_test_noisy = add_gaussian_noise(x_test, **kwargs)
    elif noise_type == 'Salt-and-Pepper':
        x_test_noisy = add_salt_and_pepper_noise(x_test, **kwargs)
    else:
        raise ValueError("Unsupported noise type. Choose 'Gaussian' or 'Salt-and-Pepper'.")

    x_test_noisy_reshaped = x_test_noisy.reshape(-1, 28, 28, 1)
    x_test_noisy_rgb = np.repeat(tf.image.resize(x_test_noisy_reshaped, (32, 32)), 3, axis=-1)

    accuracies = {}
    for model_name, model in models.items():
        if model_name == 'SVM':
            predictions = model.predict(x_test_noisy.reshape(len(x_test_noisy), -1))
        else:
            if model_name == 'ResNet':
                predictions = np.argmax(model.predict(x_test_noisy_rgb), axis=1)
            else:
                predictions = np.argmax(model.predict(x_test_noisy_reshaped), axis=1)
        accuracies[model_name] = accuracy_score(y_test, predictions)
    return accuracies, x_test_noisy

# Plot accuracy vs. noise level graph
def plot_accuracy_vs_noise(models, x_test, y_test, noise_type='Gaussian', noise_levels=None):
    if noise_levels is None:
        noise_levels = [0.01, 0.03, 0.05, 0.07, 0.1]

    accuracy_results = {model: [] for model in models.keys()}

    for noise_level in noise_levels:
        # Pass only the relevant keyword argument to evaluate_on_noisy_images
        if noise_type == 'Gaussian':
            accuracies, _ = evaluate_on_noisy_images(models, x_test, y_test, noise_type=noise_type, var=noise_level)
        elif noise_type == 'Salt-and-Pepper':
            accuracies, _ = evaluate_on_noisy_images(models, x_test, y_test, noise_type=noise_type, amount=noise_level)
        for model_name, accuracy in accuracies.items():
            accuracy_results[model_name].append(accuracy)

    plt.figure(figsize=(10, 6))
    for model_name, accuracies in accuracy_results.items():
        plt.plot(noise_levels, accuracies, label=model_name)
    plt.title(f"Accuracy vs. Noise Level ({noise_type})")
    plt.xlabel("Noise Level")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.show()

# Visualize clean vs. noisy images
def visualize_clean_vs_noisy_predictions(models, x_test, y_test, x_test_noisy, num_images=5):
    best_model = models[best_model_name]
    if best_model_name == 'SVM':
        clean_preds = best_model.predict(x_test.reshape(len(x_test), -1))
        noisy_preds = best_model.predict(x_test_noisy.reshape(len(x_test_noisy), -1))
    else:
        if best_model_name == 'ResNet':
            clean_preds = np.argmax(best_model.predict(np.repeat(tf.image.resize(x_test.reshape(-1, 28, 28, 1), (32, 32)), 3, axis=-1)), axis=1)
            noisy_preds = np.argmax(best_model.predict(np.repeat(tf.image.resize(x_test_noisy.reshape(-1, 28, 28, 1), (32, 32)), 3, axis=-1)), axis=1)
        else:
            clean_preds = np.argmax(best_model.predict(x_test.reshape(-1, 28, 28, 1)), axis=1)
            noisy_preds = np.argmax(best_model.predict(x_test_noisy.reshape(-1, 28, 28, 1)), axis=1)

    plt.figure(figsize=(15, 6))
    for i in range(num_images):
        # Clean image
        plt.subplot(2, num_images, i + 1)
        plt.imshow(x_test[i], cmap='gray')
        plt.title(f"Pred: {clean_preds[i]}\nTrue: {y_test[i]}")
        plt.axis('off')

        # Noisy image
        plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(x_test_noisy[i], cmap='gray')
        plt.title(f"Pred: {noisy_preds[i]}\nTrue: {y_test[i]}")
        plt.axis('off')

    plt.suptitle("Clean vs. Noisy Predictions", fontsize=16)
    plt.tight_layout()
    plt.show()

# Evaluate and visualize robustness
models = {'SVM': svm, 'YOLO': yolo_model, 'ResNet': resnet_model}

# Gaussian Noise
gaussian_accuracies, gaussian_noisy_images = evaluate_on_noisy_images(models, x_test_mnist, y_test_mnist, noise_type='Gaussian', var=0.05)
print("\nGaussian Noise Accuracies:")
for model_name, accuracy in gaussian_accuracies.items():
    print(f"{model_name}: {accuracy:.4f}")
visualize_clean_vs_noisy_predictions(models, x_test_mnist, y_test_mnist, gaussian_noisy_images)

# Salt-and-Pepper Noise
sp_accuracies, sp_noisy_images = evaluate_on_noisy_images(models, x_test_mnist, y_test_mnist, noise_type='Salt-and-Pepper', amount=0.05)
print("\nSalt-and-Pepper Noise Accuracies:")
for model_name, accuracy in sp_accuracies.items():
    print(f"{model_name}: {accuracy:.4f}")
visualize_clean_vs_noisy_predictions(models, x_test_mnist, y_test_mnist, sp_noisy_images)

# Plot accuracy vs. noise level for Gaussian Noise
plot_accuracy_vs_noise(models, x_test_mnist, y_test_mnist, noise_type='Gaussian')

# Plot accuracy vs. noise level for Salt-and-Pepper Noise
plot_accuracy_vs_noise(models, x_test_mnist, y_test_mnist, noise_type='Salt-and-Pepper')

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
from skimage.util import random_noise

# Function to load MNIST dataset
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

# Noise Functions
def add_gaussian_noise(images, mean=0, var=0.01):
    noisy_images = []
    for image in images:
        noise = np.random.normal(mean, var ** 0.5, image.shape)
        noisy_image = np.clip(image + noise, 0, 1)
        noisy_images.append(noisy_image)
    return np.array(noisy_images)

def add_salt_and_pepper_noise(images, amount=0.02):
    noisy_images = []
    for image in images:
        noisy_image = random_noise(image, mode='s&p', amount=amount)
        noisy_images.append(noisy_image)
    return np.array(noisy_images)

# Evaluate on Noisy Images
def evaluate_on_noisy_images(models, x_test, y_test, noise_type, **kwargs):
    if noise_type == 'Gaussian':
        x_test_noisy = add_gaussian_noise(x_test, **kwargs)
    elif noise_type == 'Salt-and-Pepper':
        x_test_noisy = add_salt_and_pepper_noise(x_test, **kwargs)
    else:
        raise ValueError("Unsupported noise type. Use 'Gaussian' or 'Salt-and-Pepper'.")

    x_test_noisy_reshaped = x_test_noisy.reshape(-1, 28, 28, 1)
    x_test_noisy_rgb = np.repeat(tf.image.resize(x_test_noisy_reshaped, (32, 32)), 3, axis=-1)

    accuracies = {}
    for model_name, model in models.items():
        if model_name == 'SVM':
            predictions = model.predict(x_test_noisy.reshape(len(x_test_noisy), -1))
        else:
            if model_name == 'ResNet':
                predictions = np.argmax(model.predict(x_test_noisy_rgb), axis=1)
            else:
                predictions = np.argmax(model.predict(x_test_noisy_reshaped), axis=1)
        accuracies[model_name] = accuracy_score(y_test, predictions)
    return accuracies, x_test_noisy

# Visualizations
def visualize_clean_vs_noisy(models, x_test, y_test, x_test_noisy, num_images=5):
    best_model = models[best_model_name]
    plt.figure(figsize=(15, 6))
    for i in range(num_images):
        plt.subplot(2, num_images, i + 1)
        plt.imshow(x_test[i], cmap='gray')
        plt.title(f"True: {y_test[i]}")
        plt.axis('off')

        plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(x_test_noisy[i], cmap='gray')
        plt.title(f"Noise Added")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Gaussian Noise
gaussian_accuracies, gaussian_noisy = evaluate_on_noisy_images(models, x_test_mnist, y_test_mnist, "Gaussian", var=0.02)

print(gaussian_accuracies)