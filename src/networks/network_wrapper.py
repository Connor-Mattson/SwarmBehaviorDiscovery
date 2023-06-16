import random
import torch
import os
import numpy as np
import cv2
from src.networks.embedding import NoveltyEmbedding
from scipy import ndimage
from src.networks.lars import LARS
from torchvision import models


def init_weights_randomly(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.uniform_(m.weight)

class NetworkWrapper:
    def __init__(self,
                 output_size=16,
                 init="Random",
                 lr=1e-3,
                 learning_decay=1.0,
                 decay_step=5,
                 margin=10.0,
                 threshold=10.0,
                 weight_decay=0,
                 new_model=False,
                 manual_schedulers=True,
                 total_epochs=100,
                 warmup=10,
                 dynamic_lr=False,
                 network=None,
                 only_features=False,
                ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if network is None:
            self.network = NoveltyEmbedding(out_size=output_size, new_model=new_model).to(self.device)
        else:
            self.network = network.to(self.device)

        self.lr = lr
        self.only_features = only_features
        self.create_optimizer()

        self.dynamic_lr = dynamic_lr

        if manual_schedulers:
            if dynamic_lr:
                # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', min_lr=2e-4,factor=0.9, patience=15, verbose=True,threshold=5e-3)
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', min_lr=2e-4, factor=0.75, patience=10, verbose=True, threshold=5e-3)
            else:
                self.scheduler = torch.optim.lr_scheduler.ChainedScheduler([
                        torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.01, total_iters=warmup),
                        torch.optim.lr_scheduler.PolynomialLR(self.optimizer, power=1.5, total_iters=(total_epochs + warmup), verbose=True)
                ])
        else:
            self.scheduler = None

        self.margin = margin
        self.loss = 0.0
        self.last_loss = 0.0
        self.learning_decay = learning_decay
        self.decay_step = decay_step
        self.scheduler_threshold = threshold
        if init == "Random":
            self.network.apply(init_weights_randomly)

    def train_batch(self, anchor, positive, negative):
        self.training_mode()
        loss_fn = torch.nn.TripletMarginLoss(margin=self.margin)
        self.optimizer.zero_grad()
        anchor_out, pos_out, neg_out = self.network.batch_network_from_numpy(anchor, positive, negative)
        loss = loss_fn(anchor_out, pos_out, neg_out)
        if loss.item() > 0 and (loss.item() - self.margin) > 0.0000001:
            loss.backward()
            if self.dynamic_lr:
                self.optimizer.step()
            else:
                self.optimizer.step()
        return loss.item()

    def eval_batch(self, anchor, positive, negative):
        loss_fn = torch.nn.TripletMarginLoss(margin=self.margin)
        anchor_out, pos_out, neg_out = self.network.batch_network_from_numpy(anchor, positive, negative)
        loss = loss_fn(anchor_out, pos_out, neg_out).item()
        return loss

    def batch_out(self, data):
        return self.network.forward_batch(data)

    # def train_triplet(self, anchor, positive=None, negative=None):
    #     self.training_mode()
    #     if positive is None:
    #         pos_images = np.stack([
    #             [ndimage.rotate(anchor, 90)],
    #             [ndimage.rotate(anchor, 180)],
    #             [ndimage.rotate(anchor, 270)],
    #         ])
    #     else:
    #         pos_images = np.array([[positive]])
    #
    #     if negative is None:
    #         raise Exception("Negative must be included in a triplet evaluation")
    #
    #     anchor_images = np.stack([[anchor] for _ in pos_images])
    #     neg_images = np.stack([[negative] for _ in pos_images])
    #
    #     losses = self.train_batch(anchor_images, pos_images, neg_images)
    #     return losses

    # def eval_triplet(self, anchor, positive=None, negative=None):
    #     self.eval_mode()
    #     if positive is None:
    #         pos_images = np.stack([
    #             [ndimage.rotate(anchor, 90)],
    #             [ndimage.rotate(anchor, 180)],
    #             [ndimage.rotate(anchor, 270)],
    #         ])
    #     else:
    #         pos_images = np.array([[positive]])
    #
    #     if negative is None:
    #         raise Exception("Negative must be included in a triplet evaluation")
    #
    #     anchor_images = np.stack([[anchor] for _ in pos_images])
    #     neg_images = np.stack([[negative] for _ in pos_images])
    #     return self.eval_batch(anchor_images, pos_images, neg_images)

    def load(self, in_folder, name):
        _dir = in_folder
        checkpoint = torch.load(os.path.join(_dir, f"{name}.pt"), map_location=self.device)
        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.network = self.network.to(self.device)

    def load_from_path(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.network = self.network.to(self.device)

    def save(self, out_folder, name):
        _dir = out_folder
        if not os.path.isdir(_dir):
            os.mkdir(_dir)
        torch.save({
            'model_state_dict': self.network.state_dict(),
        }, os.path.join(_dir, f"{name}.pt"))

    def training_mode(self):
        self.network.train()

    def eval_mode(self):
        self.network.eval()

    def create_optimizer(self, lr=None):
        if self.only_features:
            self.optimizer = torch.optim.Adam(self.network.encoder.parameters(), lr=self.lr if lr is None else lr, eps=1e-6)
        else:
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr if lr is None else lr, eps=1e-6)

    def step_scheduler(self, loss=None):
        if self.scheduler:
            if self.dynamic_lr and loss is not None:
                self.scheduler.step(loss)
            else:
                self.scheduler.step()

    def colored_input(self, image):
        """
        Embed the R,G channels of an image and then embed the entire image, concatenating the result into a vector of size 15.
        :param image: A numpy 3-channel Image
        :return: A vector of size 15 (First 5 values: R channel, Next 5 values: G channel, Last 5 values: Combined R+G channels)
        """
        NOISE = 100
        reduced_noise_r = image[:, :, 0]
        reduced_noise_g = image[:, :, 1]

        reduced_noise_r[reduced_noise_r < NOISE] = 0
        reduced_noise_g[reduced_noise_g < NOISE] = 0

        extracted_r = np.expand_dims(reduced_noise_r, axis=2)
        extracted_g = np.expand_dims(reduced_noise_g, axis=2)
        combined_r_g = np.bitwise_or(extracted_r, extracted_g)

        extracted_r = np.reshape(extracted_r, (1, 50, 50))
        extracted_g = np.reshape(extracted_g, (1, 50, 50))
        combined_r_g = np.reshape(combined_r_g, (1, 50, 50))

        r_image = np.expand_dims(extracted_r, axis=0)
        g_image = np.expand_dims(extracted_g, axis=0)
        c_image = np.expand_dims(combined_r_g, axis=0)

        r_embed = self.batch_out(r_image).detach().cpu().squeeze(dim=0).numpy()
        g_embed = self.batch_out(g_image).detach().cpu().squeeze(dim=0).numpy()
        c_embed = self.batch_out(c_image).detach().cpu().squeeze(dim=0).numpy()

        ret = np.concatenate((r_embed, g_embed, c_embed))
        return ret

    # def evaluate_lr(self, losses):
    #     if self.schedulers:
    #         self.last_losses = self.losses
    #         self.losses = losses
    #         for l in range(len(losses)):
    #             if len(self.last_losses) == 0:
    #                 continue
    #             if self.scheduler_threshold and losses[l] > self.scheduler_threshold:
    #                 continue
    #             if losses[l] < self.last_losses[l]:
    #                 continue
    #             self.schedulers[l].step()
    #         return [scheduler.get_last_lr() for scheduler in self.schedulers]
    #     return None

    def set_lr(self, lr, gamma):
        self.create_optimizer(lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.decay_step, gamma=gamma)

    def get_lr(self):
        return self.scheduler.optimizer.param_groups[0]['lr']


class ResNetWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        resnet = models.resnet18(pretrained=True)
        num_ftrs = resnet.fc.in_features
        layers_ = list(resnet.children())[:-1]
        self.resnet = torch.nn.Sequential(*layers_)
        print(f"ResNet Init with {num_ftrs}")

    def forward(self, x, greyscale=False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = x.astype(np.uint8)
        x = cv2.resize(x, dsize=(200, 200), interpolation=cv2.INTER_AREA)
        x = np.moveaxis(x, -1, 0)
        x = torch.from_numpy(x).to(device).float()

        if len(x.shape) == 3:
            x = torch.unsqueeze(x, 0)
        print(x.shape)
        if greyscale:
            x = x.repeat(1, 3, 1, 1).to(device)
        batch_size, c, h, w = x.shape
        print(f"Final Shape: {x.shape}")
        feats = self.resnet(x)
        feats_flat = torch.flatten(feats, 1)
        return feats_flat

    def batch_out(self, data, greyscale=False):
        return self.forward(data, greyscale)
