# NLP Command Classification and Model Training

This project involves training a deep learning model to classify various voice commands using image representations of audio spectrograms. The training is performed using TensorFlow and TensorFlow Hub.

## Features
- **Model**: MobileNetV2 (Transfer Learning)
- **Data**: 12 classes, 173 images
- **Method**: Data augmentation and training using `ImageDataGenerator`
- **Results**: Achieved 68.97% accuracy
- **Model Saving**: Trained model saved in `.h5` format

## Workflow

### 1. Data Loading
- Images are loaded from `/content/drive/My Drive/nlp/ZIMAGES` directory on Google Drive.
- Data is split into **80% training** and **20% validation**.

### 2. Model Training
- **Base Model**: MobileNetV2 for transfer learning.
- Output layers modified for 12 classes.
- Trained over **50 epochs**.

### 3. Visualization
- Training and validation accuracy and loss visualized using Matplotlib.

### 4. Model Saving
- The trained model is saved using:
  ```python
  model.save('/content/drive/My Drive/nlp/Modeller/snf')

