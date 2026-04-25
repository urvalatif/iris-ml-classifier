import numpy as np
import onnxruntime as ort
from sklearn.datasets import load_iris


def run_onnx_inference():
    iris = load_iris()

    onnx_model_path = "models/iris_random_forest.onnx"

    session = ort.InferenceSession(
        onnx_model_path,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name

    sample = np.array(
        [[5.1, 3.5, 1.4, 0.2]],
        dtype=np.float32
    )

    outputs = session.run(None, {input_name: sample})

    predicted_class = outputs[0][0]
    probabilities = outputs[1][0]

    print("Input sample:", sample.tolist()[0])
    print("Predicted class index:", predicted_class)
    print("Predicted flower:", iris.target_names[predicted_class])
    print("Class probabilities:", probabilities)


if __name__ == "__main__":
    run_onnx_inference()