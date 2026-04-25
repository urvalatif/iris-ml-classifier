import joblib
from sklearn.datasets import load_iris


def predict_multiple_flowers(samples):
    # Load saved model
    model = joblib.load("models/iris_model.joblib")

    # Load Iris class names
    iris = load_iris()

    # Predict multiple samples
    predictions = model.predict(samples)

    results = []

    for sample, prediction in zip(samples, predictions):
        flower_name = iris.target_names[prediction]

        results.append({
            "input": sample,
            "prediction": flower_name
        })

    return results


if __name__ == "__main__":
    # Example inputs:
    # [sepal length, sepal width, petal length, petal width]
    sample_data = [
        [5.1, 3.5, 1.4, 0.2],
        [6.2, 3.4, 5.4, 2.3],
        [5.9, 3.0, 4.2, 1.5]
    ]

    predictions = predict_multiple_flowers(sample_data)

    for item in predictions:
        print("Input:", item["input"])
        print("Predicted flower:", item["prediction"])
        print("-" * 30)