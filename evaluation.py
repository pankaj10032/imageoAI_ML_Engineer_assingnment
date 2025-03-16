import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap

# 1. Regression Metrics
def calculate_metrics(y_true, y_pred) -> Dict[str, float]:
    """
    Calculate regression metrics: MAE, RMSE, and R².
    
    Args:
        y_true: Actual target values.
        y_pred: Predicted target values.
    
    Returns:
        Dict[str, float]: Dictionary containing MAE, RMSE, and R².
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R²": r2}

# 2. Visual Evaluation
def plot_actual_vs_predicted(y_true, y_pred):
    """
    Plot a scatter plot comparing actual vs. predicted values.
    
    Args:
        y_true: Actual target values.
        y_pred: Predicted target values.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color="red", linestyle="--")  # Diagonal line
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs. Predicted Values")
    plt.show()

def plot_residuals(y_true, y_pred):
    """
    Plot residuals to identify systematic errors.
    
    Args:
        y_true: Actual target values.
        y_pred: Predicted target values.
    """
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(y=0, color="red", linestyle="--")  # Horizontal line at 0
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.show()

# 3. Interpretability with SHAP
def explain_with_shap(model, X_train, X_test):
    """
    Use SHAP to explain model predictions and summarize feature importance.
    
    Args:
        model: Trained model.
        X_train: Training features.
        X_test: Test features.
    """
    # Initialize SHAP explainer
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    
    # Summary plot of feature importance
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    
    # Force plot for a single prediction
    shap.initjs()
    shap.force_plot(explainer.expected_value, shap_values[0, :], X_test[0, :])

# Main function for evaluation
def evaluate_and_interpret(model, X_train, X_test, y_test):
    """
    Evaluate the model and interpret predictions.
    
    Args:
        model: Trained model.
        X_train: Training features.
        X_test: Test features.
        y_test: Test target values.
    """
    # Make predictions
    y_pred = model.predict(X_test).flatten()
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)
    print("Regression Metrics:", metrics)
    
    # Visual evaluation
    plot_actual_vs_predicted(y_test, y_pred)
    plot_residuals(y_test, y_pred)
    
    # Interpretability with SHAP
    explain_with_shap(model, X_train, X_test)

# Example usage in main.py
if __name__ == "__main__":
    from data_processing import load_data, preprocess_data, split_data
    from modeling import build_model, train_model
    
    # Load and preprocess data
    data = load_data("MLE-Assignment-with-Indices.csv")
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Build and train the model
    input_shape = X_train.shape[1]
    model = build_model(input_shape)
    train_model(model, X_train, y_train)
    
    # Evaluate and interpret the model
    evaluate_and_interpret(model, X_train, X_test, y_test)