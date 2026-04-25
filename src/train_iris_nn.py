import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class IrisNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )

    def forward(self, x):
        return self.network(x)


def train_iris_nn():
    iris = load_iris()

    X = iris.data.astype(np.float32)
    y = iris.target.astype(np.int64)

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    model = IrisNet()

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 200

    for epoch in range(epochs):
        model.train()

        outputs = model(X_train_tensor)
        loss = loss_function(outputs, y_train_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    model.eval()

    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        predictions = torch.argmax(test_outputs, dim=1)
        accuracy = (predictions == y_test_tensor).float().mean().item()

    print("Test accuracy:", accuracy)

    os.makedirs("models", exist_ok=True)

    torch.save(model.state_dict(), "models/iris_nn.pth")

    dummy_input = torch.randn(1, 4, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        "models/iris_nn.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=17
    )

    print("PyTorch model saved at: models/iris_nn.pth")
    print("ONNX model saved at: models/iris_nn.onnx")


if __name__ == "__main__":
    train_iris_nn()