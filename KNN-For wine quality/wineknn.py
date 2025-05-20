import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
wine = pd.read_csv('winequality-red-cleaned.csv', sep=',')

# Create three quality classes
wine['class'] = ['Low' if i <= 4 else 'Medium' if i <= 7 else 'High' for i in wine.quality]

# Drop the original quality column
wine.drop(columns=['quality'], inplace=True)

# Separate features and target
X = wine.drop('class', axis=1)
y = wine['class']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled = scaler.transform(X_test)

class KNNClassifier:
    def __init__(self, k=5):
        self.k = k
        
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = np.array(y)
        
    def predict(self, X):
        predictions = []
        for x in X:
            # Calculate distances to all training points
            distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
            
            # Get indices of k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            
            # Get labels of k nearest neighbors
            k_nearest_labels = self.y_train[k_indices]
            
            # Get most common label
            most_common = max(set(k_nearest_labels), key=list(k_nearest_labels).count)
            predictions.append(most_common)
            
        return predictions


print("Class distribution:\n", y.value_counts())
# Create and train the model
knn = KNNClassifier(k=5)
knn.fit(X_train_scaled, y_train)

# Make predictions
y_pred = knn.predict(X_test_scaled)

# Print results
print("\nClassification Report:")

print(classification_report(y_test, y_pred, zero_division=0))

# Plot class distribution
plt.figure(figsize=(10, 6))
wine['class'].value_counts().plot(kind='bar')
plt.title('Distribution of Wine Quality Classes')
plt.xlabel('Quality Class')
plt.ylabel('Count')
plt.show()

# Plot the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 6))
labels = sorted(list(set(y)))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels,
            yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

# Plot correlation matrix
correlation_matrix = X.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

print(y_train.value_counts())
