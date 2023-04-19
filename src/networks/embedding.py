import torch
import time
import torchvision.transforms as T
from scipy import ndimage
import numpy as np

class NoveltyEmbedding(torch.nn.Module):
    def __init__(self, out_size=15, new_model=False):
        super().__init__()
        if new_model:
            self.s1 = torch.nn.Sequential(
                torch.nn.Conv2d(1, 1, 3, stride=1, padding=0),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(1, 1, 3, stride=1, padding=0),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(1, 1, 3, stride=1, padding=0),
                torch.nn.LeakyReLU(),
                torch.nn.Flatten(),
                torch.nn.Linear(1936, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, out_size),
            )
        else:
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
                torch.nn.Linear(100, out_size),
            )

    def forward(self, x):
        x = self.s1(x)
        return x

    def load_model(self, file_name="cp_NAME"):
        checkpoint = torch.load(f"checkpoints/embeddings/{file_name}.pt")
        self.load_state_dict(checkpoint["model_state_dict"])

    def save_model(self):
        file_name = f"cp_{round(time.time())}"
        torch.save({
            'model_state_dict': self.state_dict(),
        }, f"checkpoints/embeddings/{file_name}.pt")

    def numpy_single_pass(self, img):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        prime = torch.from_numpy(img).to(device).float()
        out = self.forward(prime.unsqueeze(0))
        return out

    def network_from_numpy(self, anchor_img, pos_img, neg_img):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        anchor_input = torch.from_numpy(anchor_img).to(device).float()
        pos_input = torch.from_numpy(pos_img).to(device).float()
        neg_input = torch.from_numpy(neg_img).to(device).float()

        anchor_out = self.forward(anchor_input.unsqueeze(0))
        pos_out = self.forward(pos_input.unsqueeze(0))
        neg_out = self.forward(neg_input.unsqueeze(0))
        return anchor_out, pos_out, neg_out

    def batch_network_from_numpy(self, anchor_list, pos_list, neg_list):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        anchor_input = torch.from_numpy(anchor_list).to(device).float()
        pos_input = torch.from_numpy(pos_list).to(device).float()
        neg_input = torch.from_numpy(neg_list).to(device).float()

        anchor_out = self.forward(anchor_input)
        pos_out = self.forward(pos_input)
        neg_out = self.forward(neg_input)
        return anchor_out, pos_out, neg_out

    def network_with_transforms(self, anchor_img, pos_img, neg_img):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # anchor_input = torch.from_numpy(anchor_list).to(device).float().unsqueeze(0)
        # pos_input = torch.from_numpy(pos_list).to(device).float().unsqueeze(0)
        # neg_input = torch.from_numpy(neg_list).to(device).float().unsqueeze(0)

        pos_images = np.stack([
            [ndimage.rotate(pos_img, 90)],
            [ndimage.rotate(pos_img, 180)],
            [ndimage.rotate(pos_img, 270)],
            # ndimage.gaussian_filter(pos_img, 1),
            # ndimage.gaussian_filter(pos_img, 3),
            # ndimage.gaussian_filter(pos_img, 5),
        ])

        anchor_images = np.stack([[anchor_img] for _ in pos_images])
        neg_images = np.stack([[neg_img] for _ in pos_images])
        return self.batch_network_from_numpy(anchor_images, pos_images, neg_images)
