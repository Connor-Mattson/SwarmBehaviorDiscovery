import torch
import os
import numpy as np
from networks.embedding import NoveltyEmbedding
from scipy import ndimage


def init_weights_randomly(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.uniform_(m.weight)


class Ensemble:
    def __init__(self, size=3, output_size=16, init="Random", lr=1e-3, learning_decay=1.0, decay_step=5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ensemble = [
            NoveltyEmbedding(out_size=output_size).to(self.device) for _ in range(size)
        ]
        self.optimizers = [
            torch.optim.Adam(self.ensemble[i].parameters(), lr=lr) for i in range(size)
        ]
        self.schedulers = [
            torch.optim.lr_scheduler.StepLR(self.optimizers[i], step_size=decay_step, gamma=learning_decay) for i in range(size)
        ]
        if init == "Random":
            for network in self.ensemble:
                network.apply(init_weights_randomly)

    def train_batch(self, anchor, positive, negative):
        self.training_mode()
        loss_fn = torch.nn.TripletMarginLoss(margin=1.0)
        losses = []
        for i, network in enumerate(self.ensemble):
            self.optimizers[i].zero_grad()
            anchor_out, pos_out, neg_out = network.batch_network_from_numpy(anchor, positive, negative)
            loss = loss_fn(anchor_out, pos_out, neg_out)
            losses.append(loss.item())
            loss.backward()
            self.optimizers[i].step()
        return losses

    def eval_batch(self, anchor, positive, negative):
        loss_fn = torch.nn.TripletMarginLoss(margin=1.0)
        losses = []
        for i, network in enumerate(self.ensemble):
            anchor_out, pos_out, neg_out = network.batch_network_from_numpy(anchor, positive, negative)
            loss = loss_fn(anchor_out, pos_out, neg_out)
            losses.append(loss)
        return losses

    def train_triplet(self, anchor, positive=None, negative=None):
        self.training_mode()
        if positive is None:
            pos_images = np.stack([
                [ndimage.rotate(anchor, 90)],
                [ndimage.rotate(anchor, 180)],
                [ndimage.rotate(anchor, 270)],
            ])
        else:
            pos_images = np.array([[positive]])

        if negative is None:
            raise Exception("Negative must be included in a triplet evaluation")

        anchor_images = np.stack([[anchor] for _ in pos_images])
        neg_images = np.stack([[negative] for _ in pos_images])
        return self.train_batch(anchor_images, pos_images, neg_images)
        
    def eval_triplet(self, anchor, positive=None, negative=None):
        self.eval_mode()
        if positive is None:
            pos_images = np.stack([
                [ndimage.rotate(anchor, 90)],
                [ndimage.rotate(anchor, 180)],
                [ndimage.rotate(anchor, 270)],
            ])
        else:
            pos_images = [[positive]]

        if negative is None:
            raise Exception("Negative must be included in a triplet evaluation")

        anchor_images = np.stack([[anchor] for _ in pos_images])
        neg_images = np.stack([[negative] for _ in pos_images])
        return self.eval_batch(anchor_images, pos_images, neg_images)

    def binary_agreement(self, anchor, positive=None, negative=None):
        """
        If ANY disagreement occurs in the ensemble, return False, else return True
        """
        losses = self.eval_triplet(anchor, positive, negative)
        baseline = losses[0] > 0.0
        for i in range(1, len(losses)):
            if losses[i] > 0.0 != baseline:
                return False
        return True

    def entropy_agreement(self, anchor, positive=None, negative=None):
        losses = self.eval_triplet(anchor, positive, negative)
        # Softmax to get a probability distribution
        losses = torch.softmax(torch.tensor(losses), dim=0)
        entropy = -sum([l * np.log2(l) for l in losses])
        return entropy

    def load_ensemble(self, in_folder, absolute=False):
        _dir = f"checkpoints/ensembles/{in_folder}"
        for i, network in enumerate(self.ensemble):
            checkpoint = torch.load(os.path.join(_dir, f"{i}.pt"))
            network.load_state_dict(checkpoint["model_state_dict"])

    def save_ensemble(self, out_folder, absolute=False):
        _dir = f"checkpoints/ensembles/{out_folder}"
        if not os.path.isdir(_dir):
            os.mkdir(_dir)

        for i, network in enumerate(self.ensemble):
            torch.save({
                'model_state_dict': network.state_dict(),
            }, os.path.join(_dir, f"{i}.pt"))

    def training_mode(self):
        for network in self.ensemble:
            network.train()

    def eval_mode(self):
        for network in self.ensemble:
            network.eval()

    def step_schedulers(self):
        for scheduler in self.schedulers:
            scheduler.step()
