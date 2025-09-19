# Academic Stress Classification with Machine Learning

This project focuses on the application of Machine Learning (ML) models to analyze and classify data related to academic stress among students. The primary goal is to explore how different supervised learning algorithms perform when applied to a small but meaningful dataset. By training, validating, and testing multiple models, we aim to identify which approach provides the best balance between accuracy, generalization, and robustness.

The motivation behind this work is to provide insights into early detection of student stress patterns, supporting better decision-making in educational contexts.

# Objectives

- To preprocess the dataset by balancing classes with SMOTE and scaling features.

- To train and evaluate at least three supervised ML models, following the assignment requirements.

- To compare model performance across training, validation, and test splits.

- To analyze overfitting and underfitting behaviors.

- To recommend the most suitable model for deployment in a real-world educational setting.

# Models Trained

**We evaluated five models:**

- **k-Nearest Neighbors (kNN):** A distance-based classifier, optimized by testing different values of k.

- **Random Forest (RF):** An ensemble method based on decision trees, tuned with hyperparameter search and cross-validation.

- **Gradient Boosting (GB):** A boosting-based ensemble model, tested with multiple learning rates and tree depths.

- **Deep Neural Network (DNN):** A multi-layer perceptron with hidden layers (128-64-32 neurons), dropout regularization, and early stopping.

- **Logistic Regression (LogReg):** A simple but robust linear model, showing promising generalization on small datasets.

Performance was mainly measured with accuracy across train, validation, and test sets. For Random Forest, additional metrics (precision, recall, F1-score) were explored.

# Comparative Analysis

A comparative study highlighted each model’s strengths and weaknesses:

- Ensemble methods (RF, GB) achieved high training accuracy but struggled with validation performance → signs of overfitting.

- DNN underperformed due to dataset size and complexity.

- Logistic Regression provided the most balanced performance across all splits, making it the recommended candidate for production.

# Team Members

- Jean Carlo Londoño Ocampo

- Alejandro Garcés Ramírez

- María Acevedo

# Technologies Used

- Python 3.12

- Pandas, NumPy, Scikit-learn → preprocessing, classical ML models

- Imbalanced-learn (SMOTE) → dataset balancing

- TensorFlow / Keras → Deep Neural Network

- Matplotlib / Seaborn → visualization

- Joblib → model persistence

The results of this project provide valuable insights into how classical ML models and neural networks behave on real-world educational data, and serve as a foundation for building scalable solutions that can support early detection of academic stress.
