# main.py
from data_processing import load_data, preprocess_data, split_data
from modeling import build_model, train_model
from evaluation import evaluate_model

def main():
    # Load and preprocess data
    data = load_data("MLE-Assignment-with-Indices.csv")
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Build and train the model
    input_shape = X_train.shape[1]
    model = build_model(input_shape)
    history = train_model(model, X_train, y_train)
    
    # Evaluate the model
    metrics = evaluate_model(model, X_test, y_test)
    print("Model Evaluation Metrics:", metrics)

if __name__ == "__main__":
    main()