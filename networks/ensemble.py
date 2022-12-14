import random

import torch
import os
import numpy as np
from networks.embedding import NoveltyEmbedding
from scipy import ndimage


def init_weights_randomly(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.uniform_(m.weight)


class Ensemble:
    def __init__(self, size=3, output_size=16, init="Random", lr=1e-3, lr_series=None, learning_decay=1.0, decay_step=5, margin=10.0, threshold=10.0, weight_decay=0, new_model=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ensemble = [
            NoveltyEmbedding(out_size=output_size, new_model=new_model).to(self.device) for _ in range(size)
        ]

        self.lr_series = lr_series
        if lr_series is None:
            self.optimizers = [
                torch.optim.Adam(self.ensemble[i].parameters(), lr=lr, weight_decay=weight_decay) for i in range(size)
            ]
        else:
            self.optimizers = [
                torch.optim.Adam(self.ensemble[i].parameters(), lr=lr_series[i], weight_decay=weight_decay) for i in range(size)
            ]

        self.schedulers = [
            torch.optim.lr_scheduler.StepLR(self.optimizers[i], step_size=decay_step, gamma=learning_decay) for i in range(size)
        ]
        self.margin = margin
        self.losses = []
        self.last_losses = []
        self.learning_decay = learning_decay,
        self.decay_step = decay_step
        self.scheduler_threshold = threshold
        if init == "Random":
            for network in self.ensemble:
                network.apply(init_weights_randomly)

    def train_batch(self, anchor, positive, negative):
        self.training_mode()
        loss_fn = torch.nn.TripletMarginLoss(margin=self.margin)
        losses = []
        for i, network in enumerate(self.ensemble):
            self.optimizers[i].zero_grad()
            anchor_out, pos_out, neg_out = network.batch_network_from_numpy(anchor, positive, negative)
            loss = loss_fn(anchor_out, pos_out, neg_out)
            if loss.item() > 0:
                loss.backward()
                self.optimizers[i].step()
            losses.append(loss.item())
        return losses

    def eval_batch(self, anchor, positive, negative):
        loss_fn = torch.nn.TripletMarginLoss(margin=self.margin)
        losses = []
        for i, network in enumerate(self.ensemble):
            anchor_out, pos_out, neg_out = network.batch_network_from_numpy(anchor, positive, negative)
            loss = loss_fn(anchor_out, pos_out, neg_out).item()
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

        losses = self.train_batch(anchor_images, pos_images, neg_images)
        return losses
        
    def eval_triplet(self, anchor, positive=None, negative=None):
        self.eval_mode()
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

    def binary_correct(self, anchor, positive=None, negative=None):
        """
        If ALL networks agree with the triplet, return True, else return False
        """
        losses = self.eval_triplet(anchor, positive, negative)
        for i in range(0, len(losses)):
            if losses[i] > 0.0:
                return False
        return True

    def num_networks_correct(self, anchor, positive=None, negative=None):
        """
        Return the number of networks that agree with the triplet
        """
        losses = self.eval_triplet(anchor, positive, negative)
        count = 0
        for i in range(0, len(losses)):
            if losses[i] == 0.0:
                count += 1
        return count

    def entropy_agreement(self, anchor, positive=None, negative=None):
        """
        Not a good metric of entropy - Deprecate Please
        """
        losses = self.eval_triplet(anchor, positive, negative)
        # Softmax to get a probability distribution
        losses = torch.softmax(torch.tensor(losses), dim=0)
        entropy = 0
        for l in losses:
            if l < 1e-4:
                continue
            entropy += (l * (np.log(1 / l)))
        return entropy

    def majority_belief(self, anchor, positive=None, negative=None):
        """
        Return True if the majority of the ensemble believes
        """
        losses = self.eval_triplet(anchor, positive, negative)
        threshold = len(losses) / 2
        curr = 0
        for i in range(0, len(losses)):
            if losses[i] < self.margin:
                curr += 1
        return curr > threshold, losses

    def variance(self, anchor, positive=None, negative=None):
        losses = self.eval_triplet(anchor, positive, negative)
        binary_out = [0.0 if i > 0.0 else 1.0 for i in losses]
        P_t = sum(binary_out) / len(binary_out)
        return P_t * (1 - P_t)

    def load_ensemble(self, in_folder, absolute=False, full=False):
        _dir = f"checkpoints/ensembles/{in_folder}"
        if full:
            _dir = in_folder
        for i, network in enumerate(self.ensemble):
            checkpoint = torch.load(os.path.join(_dir, f"{i}.pt"))
            network.load_state_dict(checkpoint["model_state_dict"])

    def save_ensemble(self, out_folder, absolute=False, full=False):
        _dir = f"checkpoints/ensembles/{out_folder}"
        if full:
            _dir = out_folder
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
        if self.schedulers:
            for scheduler in self.schedulers:
                scheduler.step()

    def evaluate_lr(self, losses):
        if self.schedulers:
            self.last_losses = self.losses
            self.losses = losses
            for l in range(len(losses)):
                if len(self.last_losses) == 0:
                    continue
                if self.scheduler_threshold and losses[l] > self.scheduler_threshold:
                    continue
                if losses[l] < self.last_losses[l]:
                    continue
                self.schedulers[l].step()
            return [scheduler.get_last_lr() for scheduler in self.schedulers]
        return None

    def set_single_lr(self, index, lr, gamma):
        self.optimizers[index] = torch.optim.Adam(self.ensemble[index].parameters(), lr=lr)
        self.schedulers[index] = torch.optim.lr_scheduler.StepLR(self.optimizers[index], step_size=self.decay_step, gamma=gamma)

    def set_lr(self, lr, gamma):
        self.optimizers = [
            torch.optim.Adam(self.ensemble[i].parameters(), lr=lr) for i in range(len(self.ensemble))
        ]
        self.schedulers = None
        self.schedulers = [
            torch.optim.lr_scheduler.StepLR(self.optimizers[i], step_size=self.decay_step, gamma=gamma) for i in range(len(self.ensemble))
        ]

    def trim(self, index):
        self.ensemble = self.ensemble[-index:]
        self.schedulers = self.schedulers[-index:]
        self.optimizers = self.optimizers[-index:]
