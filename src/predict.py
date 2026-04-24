import joblib
from sklearn.datasets import load_iris


def predict_flower(sample):
    # Load saved model
    model = joblib.load("models/iris_model.joblib")

    # Load class names
    iris = load_iris()

    # Predict class
    prediction = model.predict([sample])

    flower_name = iris.target_names[prediction[0]]

    return flower_name


if __name__ == "__main__":
    # Example input:
    # [sepal length, sepal width, petal length, petal width]
    sample_data = [5.1, 3.5, 1.4, 0.2]

    result = predict_flower(sample_data)

    print("Predicted flower:", result)