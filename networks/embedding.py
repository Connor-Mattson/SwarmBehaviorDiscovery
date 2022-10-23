import torch
import time
import torchvision.transforms as T

class NoveltyEmbedding(torch.nn.Module):
    def __init__(self, out_size=15):
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

    def network_with_transforms(self, anchor_list, pos_list, neg_list):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        anchor_input = torch.from_numpy(anchor_list).to(device).float().unsqueeze(0)
        pos_input = torch.from_numpy(pos_list).to(device).float().unsqueeze(0)
        neg_input = torch.from_numpy(neg_list).to(device).float().unsqueeze(0)

        affine_transformer = T.Compose([
            T.ToPILImage(),
            T.RandomAffine(degrees=(0, 360), translate=(0.0, 0.02), scale=(0.9, 1.0)),
            T.ToTensor(),
            T.Normalize(0.0, 1.0),
        ])

        pos_input = affine_transformer(pos_input).to(device)

        anchor_out = self.forward(anchor_input)
        pos_out = self.forward(pos_input)
        neg_out = self.forward(neg_input)
        return anchor_out, pos_out, neg_out
