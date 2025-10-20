# Nepali Sign Language Recognizer

A real-time Nepali Sign Language character recognizer using a webcam and CNN (TensorFlow/Keras) that displays recognized gestures as text in a simple UI.

---

## Dataset

The dataset was manually collected using personal devices/webcams and currently includes **14 gesture classes**. For each gesture, approximately **1,100** raw images were captured and then augmented (flipping, rotation, translation, brightness adjustments, noise) to produce **~2,100** training images per class. A separate set of **~700** images per class was used for testing. All images are stored in **grayscale** at **300×300** pixels.

---

## Processing

Captured frames are preprocessed to extract the hand region, then converted to grayscale, resized to 300×300 pixels, and labeled. Data augmentation improves robustness and model generalization.

A CNN built with TensorFlow/Keras, consisting of multiple convolutional layers with ReLU activations and pooling, followed by flattening and dense layers with a softmax output, is trained on the prepared dataset. Training includes checkpointing and evaluation using confusion matrix, precision, recall, and F1 metrics. During deployment, webcam frames are preprocessed in the same way, fed to the trained model, and the predicted gesture is displayed as text in the UI.

---

## Results

The trained model achieved approximately **93.41% accuracy** after 10 epochs of training. Accuracy can be further improved by expanding the dataset, applying additional augmentation, tuning hyperparameters, or exploring deeper architectures and transfer learning.
