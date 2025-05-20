import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns

# Load the cleaned dataset
df = pd.read_csv('winequality-white-cleaned.csv')
df.columns = df.columns.str.strip().str.lower()

# Define features and target
X = df.drop('quality', axis=1)
y = df['quality']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=3)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("📊 Evaluation Metrics:")
print(f"MAE  = {mae:.3f}")
print(f"RMSE = {rmse:.3f}")
print(f"R²   = {r2:.3f}")

new_sample = pd.DataFrame([{
    'fixed acidity': 7.0,
    'volatile acidity': 0.27,
    'citric acid': 0.36,
    'residual sugar': 20.7,
    'chlorides': 0.045,
    'free sulfur dioxide': 45.0,
    'total sulfur dioxide': 170.0,
    'density': 1.001,
    'ph': 3.00,
    'sulphates': 0.45,
    'alcohol': 8.8
}])

# Predict wine quality
predicted_quality = model.predict(new_sample)[0]
print(f"\n🍷 Predicted wine quality: {predicted_quality:.2f}")