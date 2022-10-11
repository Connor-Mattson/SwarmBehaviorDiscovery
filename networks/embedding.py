import torch
import time


class NoveltyEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.s1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 1, 5, stride=2, padding=2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(1, 1, 3, stride=2, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(1, 1, 3, stride=2, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(3969, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 15),
        )

    def forward(self, x):
        x = self.s1(x)
        return x

    def load_model(self, file_name="cp_NAME"):
        checkpoint = torch.load(f"checkpoints/mixer/{file_name}.pt")
        self.load_state_dict(checkpoint["model_state_dict"])

    def save_model(self):
        file_name = f"cp_{round(time.time())}"
        torch.save({
            'model_state_dict': self.state_dict(),
        }, f"checkpoints/mixer/{file_name}.pt")