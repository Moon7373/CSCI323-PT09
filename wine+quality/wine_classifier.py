import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
path = 'winequality-red-cleaned.csv' 
wine_dataset = pd.read_csv(path)

# Normalize column names
wine_dataset.columns = wine_dataset.columns.str.strip().str.lower()

# Display basic info
print(wine_dataset.shape)
print(wine_dataset.head())
print(wine_dataset.isnull().sum())
print(wine_dataset.describe())

# Count of wine quality
sns.catplot(x='quality', data=wine_dataset, kind='count')

# Bar plots
plt.figure(figsize=(5, 5))
sns.barplot(x='quality', y='volatile acidity', data=wine_dataset)

plt.figure(figsize=(5, 5))
sns.barplot(x='quality', y='citric acid', data=wine_dataset)

# Correlation heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(wine_dataset.corr(), cbar=True, square=True, fmt='.1f',
            annot=True, annot_kws={'size': 8}, cmap='Blues')
plt.show()

# Features and label
X = wine_dataset.drop('quality', axis=1)
Y = wine_dataset['quality'].apply(lambda x: 1 if x >= 7 else 0)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Train model
model = RandomForestClassifier()
model.fit(X_train, Y_train)

# Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(predictions, Y_test)
print('Accuracy:', accuracy)

# Predict on sample input
input_data = (6.4, 0.31, 0.38, 2.9, 0.038, 19, 102, 0.9912, 3.17, 0.35, 11)
input_df = pd.DataFrame([input_data], columns=X.columns)
prediction = model.predict(input_df)
print("Prediction:", prediction)

if prediction[0] == 1:
    print("✅ Good quality wine")
else:
    print("❌ Bad quality wine")