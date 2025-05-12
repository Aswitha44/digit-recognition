# Handwritten Digit Recognition Ensemble

This project implements an ensemble classification model using SVM, a YOLO-inspired CNN, and ResNet50 for MNIST digit classification. It also evaluates model robustness under different types of image noise.

---

## 📄 Project Structure

**1. `main.py`**  
- Trains and evaluates three models:  
  - Support Vector Machine (SVM)  
  - YOLO-inspired Convolutional Neural Network (CNN)  
  - ResNet50  
- Combines predictions using weighted voting ensemble  
- Visualizes performance using confusion matrices and prediction plots  

**2. `noise.py`**  
- Adds **Gaussian** and **Salt-and-Pepper** noise to the MNIST dataset  
- Re-evaluates the trained models under noisy conditions  
- Compares performance to assess robustness

---

## 💻 System Requirements

- **Python Version**: 3.6 or higher  
- **Hardware**:  
  - GPU recommended for faster CNN and ResNet50 training  
  - CPU fallback supported (will be slower)

---

## 📦 Install Dependencies

Make sure the following libraries are installed:

```bash
pip install numpy tensorflow matplotlib seaborn scikit-learn scipy scikit-image opencv-python
