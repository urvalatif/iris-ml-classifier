from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os


def train_model():
    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split dataset into training and testing parts
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Create model
    model = RandomForestClassifier(random_state=42)

    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    print("Model Accuracy:", accuracy)

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/iris_model.joblib")

    print("\nModel saved successfully in models/iris_model.joblib")


if __name__ == "__main__":
    train_model()