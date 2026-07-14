import torch
import matplotlib.pyplot as plt


class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.weights = torch.nn.Parameter(
            torch.randn(
                1,
                requires_grad=True,
                dtype=torch.float
            )
        )

        self.bias = torch.nn.Parameter(
            torch.randn(
                1,
                requires_grad=True,
                dtype=torch.float
            )
        )

    def forward(self, inputs):
        return inputs * self.weights + self.bias


class TrainAndTest:
    def __init__(self, epochs):
        torch.manual_seed(42)

        self.model = LinearRegressionModel()
        self.epochs = epochs

        self.train_set_x = None
        self.train_set_y = None
        self.test_set_x = None
        self.test_set_y = None

        self.loss_func = torch.nn.L1Loss()

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=0.01
        )

        self.training_losses = []
        self.test_predictions = None
        self.test_loss = None

    def create_data(self):
        true_weight = 0.7
        true_bias = 0.3

        x = torch.linspace(
            start=0,
            end=1,
            steps=50,
            dtype=torch.float
        )

        y = true_weight * x + true_bias

        split_index = int(len(x) * 0.8)

        self.train_set_x = x[:split_index]
        self.train_set_y = y[:split_index]

        self.test_set_x = x[split_index:]
        self.test_set_y = y[split_index:]

    def train(self):
        self.model.train()

        for epoch in range(self.epochs):
            predictions = self.model(self.train_set_x)

            loss = self.loss_func(
                predictions,
                self.train_set_y
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.training_losses.append(loss.item())

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch: {epoch + 1}, "
                    f"Loss: {loss.item():.6f}"
                )

    def test(self):
        self.model.eval()

        with torch.inference_mode():
            self.test_predictions = self.model(
                self.test_set_x
            )

            self.test_loss = self.loss_func(
                self.test_predictions,
                self.test_set_y
            ).item()

        print(f"Test loss: {self.test_loss:.6f}")

        return self.test_predictions

    def visualize(self):

        figure, axes = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(13, 5)
        )

        epoch_numbers = range(
            1,
            len(self.training_losses) + 1
        )

        axes[0].plot(
            epoch_numbers,
            self.training_losses,
            color="blue",
            label="Training loss"
        )

        axes[0].set_title("Training Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("L1 Loss")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].scatter(
            self.train_set_x,
            self.train_set_y,
            color="blue",
            label="Training data"
        )

        axes[1].scatter(
            self.test_set_x,
            self.test_set_y,
            color="green",
            label="Actual test values"
        )

        axes[1].scatter(
            self.test_set_x,
            self.test_predictions,
            color="red",
            marker="x",
            s=70,
            label="Model predictions"
        )

        axes[1].plot(
            self.test_set_x,
            self.test_predictions,
            color="red",
            alpha=0.6
        )

        axes[1].set_title(
            "Actual Test Values and Model Predictions"
        )

        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        figure.tight_layout()
        plt.show()


if __name__ == "__main__":
    trainer = TrainAndTest(epochs=250)

    trainer.create_data()
    trainer.train()
    trainer.test()
    trainer.visualize()

    print(
        f"Öğrenilen weight: "
        f"{trainer.model.weights.item():.4f}"
    )

    print(
        f"Öğrenilen bias: "
        f"{trainer.model.bias.item():.4f}"
    )