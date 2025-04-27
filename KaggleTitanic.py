# -*- coding: utf-8 -*-
"""
Name: Laksh Budhrani & James Urbin
Course: CSCI B422
Assignment: Data Mining Final Project
Description: Predicting Titanic passenger survival using feature engineering 
             and machine learning models.
"""

# --- Importing necessary libraries ---
import pandas as pd  
import seaborn as sns  
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.feature_selection import f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', None)

# --- Load training and test datasets ---
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# --- Data Exploration ---
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

print("\n", train_df.info())
print("\n", test_df.info())

print("\n", train_df.head())
print("\n", test_df.head())

print("\nTraining Set:\n", train_df.isnull().sum())
print("\nTesting Set:\n", test_df.isnull().sum())

print("\nUnique train genders:", train_df['Sex'].unique())
print("Unique train cabins:", train_df['Cabin'].unique())
print("Unique train embarked:", train_df['Embarked'].unique())
print("\nUnique test genders:", test_df['Sex'].unique())
print("Unique test cabins:", test_df['Cabin'].unique())
print("Unique test embarked:", test_df['Embarked'].unique())

# --- Data Munging ---

# Fill missing values in Age with median grouped by Pclass and Sex
train_df['Age'] = train_df.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))
test_df['Age'] = test_df.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))

# Fill missing values in Embarked with mode (most common value)
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
test_df['Embarked'] = test_df['Embarked'].fillna(test_df['Embarked'].mode()[0])

# Fill missing values in Fare (test set) with median
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())

# Create new feature: Has_Cabin (1 if Cabin exists, 0 otherwise)
train_df['Has_Cabin'] = train_df['Cabin'].notnull().astype(int)
test_df['Has_Cabin'] = test_df['Cabin'].notnull().astype(int)

# Drop the original Cabin feature since it has too many unique values
train_df.drop(columns=['Cabin'], inplace=True)
test_df.drop(columns=['Cabin'], inplace=True)

# Initialize Label Encoder
le = LabelEncoder()

# Apply Label Encoding to Sex (Male = 1, Female = 0)
train_df['Sex'] = le.fit_transform(train_df['Sex'])
test_df['Sex'] = le.transform(test_df['Sex'])

# Create Family Size feature
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1

# Extract Title from Name
train_df['Title'] = train_df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
test_df['Title'] = test_df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
print("\nTitle Counts in Train Set:\n", train_df['Title'].value_counts())
print("\nTitle Counts in Test Set:\n", test_df['Title'].value_counts())

title_mapping = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
    "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare",
    "Mlle": "Miss", "Mme": "Mrs", "Countess": "Rare", "Lady": "Rare",
    "Sir": "Rare", "Capt": "Rare", "Jonkheer": "Rare", "Don": "Rare",
    "Dona": "Rare"
}

train_df['Title'] = train_df['Title'].map(title_mapping)
test_df['Title'] = test_df['Title'].map(title_mapping)

# One-Hot Encoding for Embarked
train_df = pd.get_dummies(train_df, columns=['Embarked', 'Title'], drop_first=False)
test_df = pd.get_dummies(test_df, columns=['Embarked', 'Title'], drop_first=False)

# Convert boolean columns to integers (0 or 1)
train_df[train_df.select_dtypes('bool').columns] = train_df.select_dtypes('bool').astype(int)
test_df[test_df.select_dtypes('bool').columns] = test_df.select_dtypes('bool').astype(int)

# Drop Name and Ticket since they have too many unique values
# Drop PassengerId since it holds no predictive value for survival
train_df.drop(columns=['Name', 'Ticket', "PassengerId"], inplace=True)
test_df.drop(columns=['Name', 'Ticket'], inplace=True)

print("\nTraining data after Data Munging:\n", train_df.info())
print("\nTesting data after Data Munging:\n", test_df.info())

print("\nTraining data head after Data Munging:\n", train_df.head())
print("\nTesting data head after Data Munging:\n", test_df.head())

train_df[['Age', 'Fare']].hist(figsize=(8,6), bins=30)
plt.show()

print("\nTraining data description: \n", train_df.describe())
print("\nTesting data description: \n", test_df.describe())

# Initialize scalers
min_max_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

# Apply Min-Max Scaling to Fare
train_df['Fare'] = min_max_scaler.fit_transform(train_df[['Fare']])
test_df['Fare'] = min_max_scaler.transform(test_df[['Fare']])

# Apply Standardization (Z-score Scaling) to Age
train_df['Age'] = standard_scaler.fit_transform(train_df[['Age']])
test_df['Age'] = standard_scaler.transform(test_df[['Age']])

train_df[['Age', 'Fare']].hist(figsize=(8,6), bins=30)
plt.show()

# --- Feature Selection ---

train_df_og = train_df.drop(['Survived'], axis=1)
train_df_survived = train_df['Survived']

# Compute ANOVA F-Scores and P-Values
f_scores, p_values = f_classif(train_df_og, train_df_survived)

# Create ANOVA table
anova_table = pd.DataFrame({
    'Feature': train_df_og.columns,
    'F-Score': f_scores,
    'P-Value': p_values
})

# Sort by F-Score (higher = more significant)
anova_table = anova_table.sort_values(by='F-Score', ascending=False)

print("\nANOVA Table for Titanic Dataset:")
print(anova_table)

X = train_df.drop(columns=['Survived'])  # Features
y = train_df['Survived']  # Target variable

# --- Feature Selection using ElasticNet ---
enet_model = ElasticNet(alpha=0.1, l1_ratio=0.5)
enet_model.fit(X, y)

# Get all coefficients (even zeroes)
coef = enet_model.coef_

# Create a DataFrame for plotting
elasticnet_features_all = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': coef
}).sort_values(by='Coefficient', key=abs, ascending=False)

# Plot ALL feature importances
plt.figure(figsize=(12, 8))
sns.barplot(data=elasticnet_features_all, x='Coefficient', y='Feature', hue='Feature', palette='coolwarm', legend=False)
plt.title('ElasticNet Coefficients for All Features')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# Optional: still use only non-zero ones for modeling
selected_features = elasticnet_features_all[elasticnet_features_all['Coefficient'] != 0]['Feature'].values
print("\nSelected Features (Non-Zero Coefficients):", list(selected_features))

sns.barplot(x='Pclass', y='Survived', data=train_df)
plt.title('Survival Rate by Passenger Class')
plt.show()

sns.countplot(x=train_df['Sex'].map({0: 'Female', 1: 'Male'}), hue=train_df['Survived'].map({0: 'Did Not Survive', 1: 'Survived'}))

plt.title('Survival by Gender')
plt.xlabel("Gender")
plt.ylabel("Count")
plt.legend(title="Survival")
plt.show()

sns.barplot(x='Title_Mr', y='Survived', data=train_df)
plt.title("Survival Rate for Passengers with 'Mr' Title")
plt.xlabel("Is Title_Mr")
plt.ylabel("Survival Rate")
plt.show()

# --- Model Selection and Analysis ---

# Define different classification models
estimators = {
    'LogisticRegression': LogisticRegression(),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'SVC': SVC(gamma='scale'),
    'GaussianNB': GaussianNB(),
    'DecisionTree': DecisionTreeClassifier(),
    'RandomForest': RandomForestClassifier(n_estimators=100)
}

# Define KFold cross-validation
kfold = KFold(n_splits=10, random_state=11, shuffle=True)

# Iterate through classifiers and compute cross-validation scores
for estimator_name, estimator_object in estimators.items():
    scores = cross_val_score(estimator=estimator_object, X=train_df[selected_features], y=train_df['Survived'], cv=kfold)
    
    print(f'{estimator_name:>20}: '
          f'mean accuracy={scores.mean():.2%}; '
          f'standard deviation={scores.std():.2%}')
    

# Initialize an empty dictionary to store predictions
test_predictions = {}

# Iterate through models and generate predictions on `test_df`
for estimator_name, estimator_object in estimators.items():
    # Train the model on training data
    model = estimator_object.fit(train_df[selected_features], train_df['Survived'])
    
    # Predict survival for test data
    test_predictions[estimator_name] = model.predict(test_df[selected_features])

    # Save predictions as CSV
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': test_predictions[estimator_name]
    })
    
    # File name based on model name
    filename = f"submission_{estimator_name}.csv"
    
    # Save CSV file
    submission.to_csv(filename, index=False)
    
    print(f"✅ Saved predictions for {estimator_name} as {filename}")

# --- Keras Neural Network Model ---

import os
os.environ["KERAS_BACKEND"] = "torch"
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense,Dropout
from keras.models import Sequential
from keras.datasets.mnist import load_data
from keras.utils import to_categorical

# One-hot encode target for NN
y_nn = to_categorical(train_df['Survived'], num_classes=2)

# Train/test split
xTrain, xTest, yTrain, yTest = train_test_split(
    train_df[selected_features], y_nn, 
    test_size=0.25, random_state=42, stratify=train_df['Survived']
)

# Define model
nn_model = Sequential()
nn_model.add(Input(shape=(xTrain.shape[1],)))
nn_model.add(Dense(128, activation="relu"))
nn_model.add(Dropout(0.3))
nn_model.add(Dense(64, activation="relu"))
nn_model.add(Dropout(0.2))
nn_model.add(Dense(32, activation="relu"))
nn_model.add(Dense(2, activation="softmax"))

# Compile model
nn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
nn_model.fit(xTrain, yTrain, epochs=100, batch_size=32, verbose=1)

# Evaluate model
loss, accuracy = nn_model.evaluate(xTest, yTest)
print(f"[Neural Network] Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

# Save model
nn_model.save("titanic_nn_model.keras")

# --- Predict on Test Set ---
X_submission = test_df[selected_features]
pred_probs = nn_model.predict(X_submission)
preds = np.argmax(pred_probs, axis=1)

# Submission file
submission_nn = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': preds
})
submission_nn.to_csv("submission_NeuralNet.csv", index=False)
print("✅ Neural Net predictions saved to submission_NeuralNet.csv")
  