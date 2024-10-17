import os
import random
import time
import torch
import numpy as np
from torchvision.models import resnet18
from torchvision.transforms import Resize
from torch import nn, tensor, is_tensor
from data.swarmset import SwarmDataset
from scipy import ndimage
import matplotlib.pyplot as plt
from src.networks.transforms import crop_numpy
from src.networks.lars import LARS
from src.eval.tsne import embed_tsne

PARAMS = {
    "TRAINING_DATA": "src/z-experiments/Sum24-notebooks/data/HeRo-Pro",
    "TESTING_DATA": "src/z-experiments/Sum24-notebooks/data/HeRo-Val",
    "VALIDATION_TRIPLETS": "src/z-experiments/Sum24-notebooks/data/oracle_labels/val_data.txt",
    "VALIDATION_LABELS": "src/z-experiments/Sum24-notebooks/data/oracle_labels/HeRoVal.csv",
}

class Encoder(nn.Module):
    def __init__(self, repr_size=512, device="cpu", resnet_backbone=True):
        super(Encoder, self).__init__()
        self.repr_size = repr_size
        self.device = device

        if resnet_backbone:
            res = resnet18(pretrained=True).to(self.device)
            resnet_feature_layers = list(res.children())[:-1]
            self.backbone = nn.Sequential(*resnet_feature_layers)
        else:
            self.backbone = self.get_homemade_backbone()

    def forward(self, x):
        x = self.backbone(x)
        return x

class Projection(nn.Module):
    def __init__(self, repr_size=512, proj_size=64, hidden_size=128, device="cpu"):
        super(Projection, self).__init__()
        self.proj_size = proj_size
        self.hidden_size = hidden_size
        self.repr_size = repr_size
        self.device = device
        self.projection = torch.nn.Sequential(
            nn.Linear(self.repr_size, self.hidden_size, bias=False),
            nn.BatchNorm1d(self.hidden_size), # Add BN
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.proj_size),
        ).to(self.device)

    def forward(self, x):
        return self.projection(x)


class SimCLR(nn.Module):
    def __init__(self, repr_size=512, proj_size=64, hidden_size=128, temperature=1.0, resnet_backbone=True):
        super(SimCLR, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else "mps" if torch.has_mps else 'cpu')
        self.device = "cpu"
        self.repr_size = repr_size
        self.proj_size = proj_size
        self.hidden_size = hidden_size
        self.tau = temperature
        self.backbone = Encoder(repr_size, device=self.device, resnet_backbone=resnet_backbone)

        # The projection network dynamically updates at runtime
        self.projection = None
        self.reset_projection()

    def reset_projection(self):
        self.projection = Projection(repr_size=self.repr_size, proj_size=self.proj_size, hidden_size=self.hidden_size, device=self.device)

    def preprocess(self, x):
        """
        Performs preprocessing on the input before feeding into the model.
        Args:
            x: Numpy or Tensor
        Returns: A Tensor
        """
        # Convert Numpy to Tensor
        if not is_tensor(x):
            x = torch.tensor(x, device=self.device).float()
            if len(x.shape) == 3:
                x = x.unsqueeze(0)

        # Resizing (TorchVision)
        x = Resize((64, 64))(x)
        return x

    def encode(self, x):
        x = self.preprocess(x)
        return self.backbone(x)

    def project(self, x):
        return self.projection(x)

    def forward(self, x):
        x = self.preprocess(x)
        enc = self.backbone(x)
        enc = torch.flatten(enc, start_dim=1)
        return self.project(enc)

    def get_homemade_backbone(self):
        return torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 5, stride=2, padding=2),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.Conv2d(64, 64, 3, stride=3, padding=1),
            torch.nn.MaxPool2d(2, stride=1),
            torch.nn.Flatten(),
            torch.nn.Linear(1600, self.repr_size),
        )

    def batch_pair_loss(self, batch):
        """
        Computes the loss for a batch of data.
        This function follows the implementation of SimCLR, which represents pairs of inputs linearly via indices i and (i + 1).
        It is a bit confusing to interpret input in this way, please refer to Algorithm 1 in the paper for the 1-to-1 correspondence with this code.

        Args:
            batch: A (2N x 64 x 64) Tensor containing N (i, j) positive pairs of images.
        Returns:
            loss: The average NX-ent (Normalized Cross-Entropy) Loss between the pairs in the batch.
        """

        # TODO: Normalize?
        emb_batch = self.forward(batch)

        # For this to work, the size of the batch should be an even number (pair inputs)
        assert len(emb_batch) % 2 == 0
        N = len(emb_batch) // 2

        # Calculate the Similarity Matrix using Cosine Similarity (Algo ln 14-16)
        sim = torch.zeros((2*N, 2*N), device=self.device)
        for i in range(2 * N):
            x_i = emb_batch[i].repeat(2 * N, 1)  # Create an ((2N - 2) x M) tensor of x_i's, (M = self.proj_size)
            similarity = self.similarity(x_i, emb_batch)
            sim[i] = similarity

        # Apply Temperature
        sim /= self.tau

        # Calculate Loss using the Similarity Matrix
        loss = 0
        for k in range(N):
            loss += self.pair_loss(sim, (2*k), (2*k) + 1)  # Loss for pair (i, j)
            loss += self.pair_loss(sim, (2 * k) + 1, (2 * k))  # Loss for pair (j, i)
        loss *= 1 / (2 * N)
        return loss

    def pair_loss(self, sim, i, j):
        l_func = nn.CrossEntropyLoss(reduction='mean', ignore_index=i)  # Ignore index here prevents the summing where i==k, per algo
        target = torch.tensor(j, device=self.device)
        inp = sim[i]
        l = l_func(inp, target)
        return l

    def similarity(self, x1, x2):
        return torch.nn.CosineSimilarity(dim=1, eps=1e-6)(x1, x2)

    def save(self):
        torch.save(self.state_dict(), os.path.join(f"{int(time.time())}.pt"))

    @staticmethod
    def load(path):
        net=SimCLR(proj_size=128)
        net.load_state_dict(torch.load(path))
        return net

class Trainer:
    def __init__(self, dataset, val_ds, batch_size=32, max_iterations=100):
        self.dataset = dataset
        self.val_dataset = val_ds
        self.val_labels = np.loadtxt(PARAMS["VALIDATION_TRIPLETS"], dtype=int)
        self.class_labels = np.loadtxt(PARAMS["VALIDATION_LABELS"], dtype=int, delimiter=",")
        self.batch_size = batch_size
        self.model = SimCLR(repr_size=512, proj_size=128, temperature=0.1, resnet_backbone=True)
        # self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.optim = LARS(self.model.parameters(), lr=1.17, weight_decay=1.5e-7)
        self.max_iters = max_iterations

    def train(self):
        training_losses = []
        project_accuracy, encode_accuracy = [], []
        self.model.train()
        self.tsne_embed()
        try:
            for i in range(self.max_iters):
                train_loss = self.run_epoch()
                encode_acc, project_acc = self.run_accuracy()

                print(f"Epoch {i} complete! Train Loss: {train_loss:.4f}, Encoder Accuracy: {encode_acc:.4f}, Projection Accuracy: {project_acc:.4f}")
                training_losses.append(train_loss)
                project_accuracy.append(project_acc)
                encode_accuracy.append(encode_acc)
                self.tsne_embed()
        except KeyboardInterrupt:
            print("Training Loop was interrupted!")
        self.model.save()
        self.report(training_losses, encode_accuracy, project_accuracy)

    def run_epoch(self):
        losses = []
        self.dataset.shuffle()
        for i in range(0, len(self.dataset), self.batch_size):
            self.optim.zero_grad()
            positives_a = np.array([
                crop_numpy(self.dataset.shuff_sample(i + j)[0][::3], size=(64, 64))
                for j in range(self.batch_size)
            ])
            positives_b = np.array([
                ndimage.rotate(self.dataset.shuff_sample(i + j)[0][::3], random.choice([90, 180, 270]), axes=(1, 2))
                for j in range(self.batch_size)
            ])

            # We shape the input into the format of the SimCLR Paper
            np_a = np.expand_dims(positives_a, axis=1)
            np_b = np.expand_dims(positives_b, axis=1)
            h_concat = np.hstack((np_a, np_b))
            h_reshape = np.reshape(h_concat, (self.batch_size * 2, 3, 64, 64))

            # Forward Pass through the network
            loss = self.model.batch_pair_loss(h_reshape)
            loss.backward()
            self.optim.step()

            # Record information for data analysis
            losses.append(loss.item())
        return (sum(losses) / len(losses))

    def run_accuracy(self):
        self.model.eval()
        with torch.no_grad():
            encode_correct, project_correct, total = 0, 0, 0
            for triple in self.val_labels:
                x = self.val_dataset[triple[0]][0][::3]
                y = self.val_dataset[triple[1]][0][::3]
                z = self.val_dataset[triple[2]][0][::3]
                encodeX = self.model.encode(x)
                encodeY = self.model.encode(y)
                encodeZ = self.model.encode(z)

                projectX = self.model.project(torch.flatten(encodeX, start_dim=1))
                projectY = self.model.project(torch.flatten(encodeY, start_dim=1))
                projectZ = self.model.project(torch.flatten(encodeZ, start_dim=1))

                similarityXY = self.model.similarity(encodeX, encodeY)
                similarityXZ = self.model.similarity(encodeX, encodeZ)
                if similarityXY > similarityXZ:
                    encode_correct += 1

                similarityXY = self.model.similarity(projectX, projectY)
                similarityXZ = self.model.similarity(projectX, projectZ)
                if similarityXY > similarityXZ:
                    project_correct += 1

                total += 1
        return (encode_correct / total) * 100, (project_correct / total) * 100

    def report(self, training_loss, encoder_acc, project_acc):
        x = range(len(training_loss))
        plt.plot(x, training_loss, label="Training Loss")
        plt.title("Similarity Learning Training Loss")
        plt.xlabel("Iterations")
        plt.ylabel("NT-Xent Loss")
        plt.show()

        plt.plot(x, encoder_acc, label="Testing Accuracy")
        plt.title("Similarity Learning Encoder Accuracy")
        plt.xlabel("Iterations")
        plt.ylabel("Testing Accuracy (%)")
        plt.show()

        plt.plot(x, project_acc, label="Testing Accuracy")
        plt.title("Similarity Learning Projection Accuracy")
        plt.xlabel("Iterations")
        plt.ylabel("Testing Accuracy (%)")
        plt.show()

        path = str(int(time.time()))
        os.mkdir(path)
        np.savetxt(os.path.join(path, "training_loss.txt"), training_loss)
        np.savetxt(os.path.join(path, "encoder_accuracy.txt"), encoder_acc)
        np.savetxt(os.path.join(path, "projection_accuracy.txt"), project_acc)

    def tsne_embed(self):
        self.model.eval()
        with torch.no_grad():
            embeddings = []
            for i in range(len(self.val_dataset)):
                x = self.val_dataset[i][0][::3]
                embedding = self.model.encode(x)
                embedding = embedding.squeeze(3).squeeze(2).squeeze(0)
                embeddings.append(embedding.detach().cpu().tolist())
            e = np.array(embeddings)
            embed_tsne(e, self.class_labels[:, -1].tolist())

if __name__ == "__main__":
    ds = SwarmDataset(PARAMS["TRAINING_DATA"], rank=0, video=True)
    val_ds = SwarmDataset(PARAMS["TESTING_DATA"], rank=0, video=True)

    trainer = Trainer(ds, val_ds, batch_size=1000, max_iterations=100)
    trainer.train()

