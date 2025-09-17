# Workshop2_AJM

This project focuses on the application of Machine Learning (ML) models to analyze and classify data related to academic stress among students. The primary objective is to build, evaluate, and compare different supervised learning algorithms to determine which approach provides the best balance between accuracy, generalization, and robustness.

The models evaluated include:

- Random Forest (RF): An ensemble method based on decision trees, optimized through hyperparameter tuning and cross-validation.

- K-Nearest Neighbors (KNN): A distance-based classifier tested with multiple values of k to identify the optimal configuration.

- Deep Neural Network (DNN): A multi-layer perceptron with hidden layers (128-64-32 neurons), dropout regularization, and early stopping to prevent overfitting.

- And others

Each model was trained and validated on preprocessed datasets, with performance measured across train, validation, and test sets using key metrics such as accuracy, precision, recall, and F1-score. (Just for RF, the rest only shows the accuracy)

In addition to individual performance, a comparative analysis was conducted to identify strengths and weaknesses of each approach, focusing on overfitting risks, generalization gaps, and the feasibility of deployment.

The results of this study provide insights into how classical ML models and neural networks behave on real-world educational data, and serve as a foundation for building scalable solutions that can support early detection of student stress patterns. Enjoy :) 
