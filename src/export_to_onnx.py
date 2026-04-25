import os
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def export_model_to_onnx():
    iris = load_iris()

    X = iris.data.astype(np.float32)
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Sklearn model accuracy:", accuracy)

    os.makedirs("models", exist_ok=True)

    initial_type = [
        ("float_input", FloatTensorType([None, 4]))
    ]

    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        options={id(model): {"zipmap": False}}
    )

    onnx_path = "models/iris_random_forest.onnx"

    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"ONNX model saved at: {onnx_path}")


if __name__ == "__main__":
    export_model_to_onnx()