# Parkinson's Disease Diagnosis Using Sketch Analysis and Deep Learning

This research project focuses on the detection of Parkinson's Disease (PD) by analyzing sketching behavior, a fine motor symptom of PD. We conducted experiments on a substantial number of individuals, both with PD and a healthy control group. The objective was to introduce a system capable of differentiating PD patients based on their sketching behavior.

## Methodology

We utilized deep learning algorithms, particularly Convolutional Neural Networks (CNNs), to classify sketched images and distinguish individuals affected by Parkinson's Disease from those in the healthy control group. The study employed various CNN models, incorporating transfer learning techniques and applying them to Spiral and Wave sketched data.

### Preprocessing

Preprocessing is crucial for training deep learning models, as it helps the model generalize better to new data, and can significantly improve the model’s performance. Four preprocessing methods and algorithms were performed on the dataset:

1. Keras’ ImageDataGenerator: Used for performing image augmentation and preprocessing.
2. Zhang-Suen Thinning Algorithm: Implemented to reduce the width of an object in an image to a single pixel width.
3. Data Blurring: A Gaussian blur was applied to some of the images as a data augmentation technique.
4. Otsu Thresholding: Image thresholding was performed using Otsu’s method.

### Transfer Learning

Transfer learning is a machine learning technique that leverages knowledge gained from training a model on one task and applies it to a different, but related, task. It is particularly valuable when working with limited data for a new task, as the pre-trained model has already learned useful features from a different but related domain.

### Models Used

1. VGG-16: A CNN architecture developed by Oxford that consists of 13 convolutional layers and 3 fully connected layers.
2. ResNet50: A deep learning architecture within the ResNet family known for introducing the concept of residual learning.
3. DenseNet: Architectures consist of densely connected blocks, which include convolutional, batch normalization, and activation layers.
4. Xception: Based on the concept of depthwise separable convolutions.

## Results and Conclusion

The accuracies of the models VGG-16, and ResNet50 were estimated by tuning different hyperparameters and figuring out various gains and losses of these models on spiral and wave sketching image data. The VGG-16 model outperformed the ResNet model in terms of accuracy for spiral sketching. This study found that the VGG-16 model for learning rate 1.0e-5, which provides 86.67% accuracy, is the best transfer model for the problem of Parkinson's Detection.
