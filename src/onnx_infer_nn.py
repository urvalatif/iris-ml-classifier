import numpy as np
import onnxruntime as ort


def run_nn_onnx_inference():
    session = ort.InferenceSession(
        "models/iris_nn.onnx",
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name

    sample = np.array(
        [[0.0, 0.0, 0.0, 0.0]],
        dtype=np.float32
    )

    outputs = session.run(None, {input_name: sample})

    logits = outputs[0]
    predicted_class = int(np.argmax(logits, axis=1)[0])

    class_names = ["setosa", "versicolor", "virginica"]

    print("Predicted class index:", predicted_class)
    print("Predicted flower:", class_names[predicted_class])
    print("Raw model output:", logits)


if __name__ == "__main__":
    run_nn_onnx_inference()