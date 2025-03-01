 Parkinson's Disease Detection using Machine Learning
This project aims to detect Parkinson's Disease using various machine learning algorithms. By utilizing features extracted from biomedical voice measurements, we predict whether a person is suffering from Parkinson's Disease or not. The dataset used contains multiple attributes related to speech processing, and the target variable is status, which indicates the presence of Parkinson's Disease.

Project Overview
Parkinson’s Disease is a neurological disorder that affects movement, causing tremors, stiffness, and difficulty with walking and coordination. Early detection is crucial for management and improving the quality of life for patients. In this project, we leverage machine learning algorithms to identify patterns in voice features that can help detect the disease.

Key Features of the Project:
Data Preprocessing: Handling missing data, scaling, and normalizing features.
Exploratory Data Analysis: Statistical summaries, correlation heatmaps, and visualizations.
Model Training: Training multiple machine learning models such as Logistic Regression, Random Forest, Support Vector Machine, etc.
Model Evaluation: Evaluating models based on accuracy, precision, recall, and F1-score.
Dataset
The dataset used in this project contains biomedical voice measurements from people with and without Parkinson's Disease. It was sourced from the UCI Machine Learning Repository and contains the following attributes:

MDVP:Fo(Hz): Average vocal fundamental frequency.
MDVP:Fhi(Hz): Maximum vocal fundamental frequency.
MDVP:Flo(Hz): Minimum vocal fundamental frequency.
MDVP:Jitter(%), MDVP:Jitter(Abs), etc.: Several measures of variation in fundamental frequency.
status: The target variable (0 - healthy, 1 - Parkinson's Disease).
You can download the dataset here.

Methodology
Data Preprocessing:

Dropping irrelevant columns (if any) and handling missing data.
Feature scaling using StandardScaler.
Splitting the dataset into training and testing sets.
Exploratory Data Analysis (EDA):

Analyzing the distribution of features.
Visualizing the relationship between features and the target variable (status).
Generating correlation matrices and heatmaps.
Model Building:

We trained multiple machine learning models:
Logistic Regression
Random Forest Classifier
K-Nearest Neighbors (KNN)
Support Vector Machine (SVM)
Hyperparameter tuning using GridSearchCV.
Training on the training set and evaluating on the test set.
Model Evaluation:

Accuracy score
Confusion matrix, precision, recall, and F1-score.
ROC Curve and AUC scores.
Results
After training and evaluating various models, the Random Forest Classifier achieved the best results with high accuracy, precision, and recall scores. The ROC Curve indicates the model's ability to discriminate between healthy individuals and those with Parkinson's Disease.

Conclusion
This project demonstrates the effective use of machine learning in detecting Parkinson’s Disease using voice measurements. Early detection models like these can potentially be used to develop diagnostic tools for medical professionals.
