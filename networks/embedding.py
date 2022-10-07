import torch
import time


class NoveltyEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.s1 = torch.nn.Sequential(
            torch.nn.Linear(256, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 60),
            torch.nn.ReLU(),
            torch.nn.Linear(60, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 10),
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