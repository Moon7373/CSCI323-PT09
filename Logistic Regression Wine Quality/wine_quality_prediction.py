#!/usr/bin/env python3
# Wine Quality Prediction - Logistic Regression
# Using red and white wine datasets

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set style for plots
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

# Function to map quality scores into categories
def map_quality(x):
    """
    Maps wine quality scores into categories:
    0 = low (quality <= 4)
    1 = medium (5 <= quality <= 7)
    2 = high (quality >= 8)
    """
    if x <= 4:
        return 0  # low
    elif 5 <= x <= 7:
        return 1  # medium
    else:
        return 2  # high

# Function to load and preprocess data
def load_and_preprocess_data(filepath):
    """
    Loads the wine dataset, adds quality categories and returns processed dataframe
    """
    # Load the dataset
    df = pd.read_csv(filepath)
    
    # Add quality category column
    df['quality_category'] = df['quality'].apply(map_quality)
    
    # Display basic information
    print(f"Dataset shape: {df.shape}")
    print(f"Quality distribution: \n{df['quality_category'].value_counts()}")
    
    return df

# Function for exploratory data analysis
def perform_eda(df, wine_type):
    """
    Performs exploratory data analysis on the wine dataset
    """
    print(f"\n--- Exploratory Data Analysis for {wine_type} Wine ---")
    
    # Basic statistics
    print("\nBasic Statistics:")
    print(df.describe())
    
    # Correlation matrix
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f'Correlation Matrix for {wine_type} Wine')
    plt.tight_layout()
    plt.savefig(f'{wine_type}_correlation_matrix.png')
    
    # Distribution of quality categories
    plt.figure(figsize=(8, 6))
    sns.countplot(x='quality_category', data=df, hue='quality_category', legend=False, palette='viridis')
    plt.title(f'Distribution of Quality Categories for {wine_type} Wine')
    plt.xlabel('Quality Category (0=Low, 1=Medium, 2=High)')
    plt.ylabel('Count')
    plt.savefig(f'{wine_type}_quality_distribution.png')
    
    # Feature importance - correlation with quality
    plt.figure(figsize=(10, 8))
    feature_importance = correlation_matrix['quality_category'].drop('quality_category').sort_values(ascending=False)
    sns.barplot(x=feature_importance.values, y=feature_importance.index, hue=feature_importance.index, legend=False, palette='viridis')
    plt.title(f'Feature Correlation with Quality for {wine_type} Wine')
    plt.xlabel('Correlation Coefficient')
    plt.tight_layout()
    plt.savefig(f'{wine_type}_feature_importance.png')

# Function to train and evaluate model
def train_evaluate_model(df, wine_type):
    """
    Trains a logistic regression model and evaluates its performance
    """
    print(f"\n--- Model Training and Evaluation for {wine_type} Wine ---")
    
    # Separate features and target
    X = df.drop(['quality', 'quality_category'], axis=1)
    y = df['quality_category']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train logistic regression model
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, zero_division=0)
    print(report)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Low', 'Medium', 'High'],
                yticklabels=['Low', 'Medium', 'High'])
    plt.title(f'Confusion Matrix for {wine_type} Wine')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f'{wine_type}_confusion_matrix.png')
    
    # Feature importance
    plt.figure(figsize=(10, 8))
    coefficients = pd.DataFrame(
        model.coef_.mean(axis=0), 
        index=X.columns, 
        columns=['Coefficient']
    ).sort_values(by='Coefficient', ascending=False)
    
    sns.barplot(x=coefficients['Coefficient'], y=coefficients.index, hue=coefficients.index, legend=False, palette='viridis')
    plt.title(f'Feature Importance for {wine_type} Wine')
    plt.xlabel('Coefficient Magnitude')
    plt.tight_layout()
    plt.savefig(f'{wine_type}_feature_importance_model.png')
    
    return model, scaler

# Function to make predictions on new data
def make_predictions(model, scaler, features, wine_type):
    """
    Makes predictions on new wine samples
    """
    print(f"\n--- Making predictions for new {wine_type} wine samples ---")
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make predictions
    predictions = model.predict(features_scaled)
    
    # Convert numerical predictions to categories
    categories = ["Low", "Medium", "High"]
    prediction_labels = [categories[pred] for pred in predictions]
    
    # Display results
    result_df = pd.DataFrame(features)
    result_df['Predicted Quality'] = prediction_labels
    print(result_df)
    
    return prediction_labels

# Main function
def main():
    print("--- Wine Quality Prediction using Logistic Regression ---\n")
    
    # Process Red Wine
    print("\n=== RED WINE DATASET ===")
    red_wine_df = load_and_preprocess_data('winequality-red-cleaned.csv')
    perform_eda(red_wine_df, 'Red')
    red_model, red_scaler = train_evaluate_model(red_wine_df, 'Red')
    
    # Process White Wine
    print("\n=== WHITE WINE DATASET ===")
    white_wine_df = load_and_preprocess_data('winequality-white-cleaned.csv')
    perform_eda(white_wine_df, 'White')
    white_model, white_scaler = train_evaluate_model(white_wine_df, 'White')
    
    # Example: Make predictions on sample data
    print("\n=== SAMPLE PREDICTIONS ===")
    
    # Sample red wine data (adjust features based on your dataset)
    red_sample = pd.DataFrame({
        'fixed acidity': [7.4, 8.1],
        'volatile acidity': [0.7, 0.38], 
        'citric acid': [0.0, 0.28],
        'residual sugar': [1.9, 2.1],
        'chlorides': [0.076, 0.066],
        'free sulfur dioxide': [11.0, 13.0],
        'total sulfur dioxide': [34.0, 30.0],
        'density': [0.9978, 0.9968],
        'ph': [3.51, 3.23],
        'sulphates': [0.56, 0.73],
        'alcohol': [9.4, 9.7]
    })
    make_predictions(red_model, red_scaler, red_sample, 'Red')
    
    # Sample white wine data (adjust features based on your dataset)
    white_sample = pd.DataFrame({
        'fixed acidity': [7.0, 6.3],
        'volatile acidity': [0.27, 0.3], 
        'citric acid': [0.36, 0.34],
        'residual sugar': [20.7, 1.6],
        'chlorides': [0.045, 0.049],
        'free sulfur dioxide': [45.0, 14.0],
        'total sulfur dioxide': [170.0, 132.0],
        'density': [1.001, 0.994],
        'ph': [3.0, 3.3],
        'sulphates': [0.45, 0.49],
        'alcohol': [8.8, 9.5]
    })
    make_predictions(white_model, white_scaler, white_sample, 'White')

if __name__ == "__main__":
    main() 