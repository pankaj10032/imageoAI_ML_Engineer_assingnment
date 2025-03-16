# modeling.py
from typing import Dict, Any
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_model(input_shape: int, hidden_units: int = 64, dropout_rate: float = 0.2, learning_rate: float = 0.001) -> Sequential:
    """
    Build a simple feedforward neural network.
    
    Args:
        input_shape (int): Number of input features.
        hidden_units (int): Number of units in the hidden layer.
        dropout_rate (float): Dropout rate for regularization.
        learning_rate (float): Learning rate for the optimizer.
    
    Returns:
        Sequential: Compiled Keras model.
    """
    model = Sequential([
        Dense(hidden_units, activation="relu", input_shape=(input_shape,)),
        Dropout(dropout_rate),
        Dense(hidden_units // 2, activation="relu"),
        Dense(hidden_units // 4, activation="relu"),
        Dense(1)  # Output layer for regression
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
    return model

def train_model(model: Sequential, X_train, y_train, epochs: int = 50, batch_size: int = 32, validation_split: float = 0.2) -> Any:
    """
    Train the model.
    
    Args:
        model (Sequential): The Keras model.
        X_train: Training features.
        y_train: Training target.
        epochs (int): Number of epochs.
        batch_size (int): Batch size.
        validation_split (float): Proportion of training data to use for validation.
    
    Returns:
        Any: Training history.
    """
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=1)
    return history