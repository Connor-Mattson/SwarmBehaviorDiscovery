import random

import torch
import cv2
import matplotlib
import matplotlib.pyplot as plot
import numpy as np
import os
import time
import torchvision
from torchvision import datasets, transforms
from generation.halted_evolution import HaltedEvolution

class BehaviorAutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(200*200, 8192),
            torch.nn.ReLU(),
            torch.nn.Linear(8192, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 256)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(256, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 2048),
            # torch.nn.Dropout(0.01),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 8192),
            torch.nn.ReLU(),
            torch.nn.Linear(8192, 200 * 200),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encoded(self, x):
        encoded = self.encoder(x)
        return encoded

    def load_model(self, file_name="cp_NAME"):
        checkpoint = torch.load(f"checkpoints/encoder/{file_name}.pt")
        self.load_state_dict(checkpoint["model_state_dict"])

    def save_model(self):
        file_name = f"cp_{round(time.time())}"
        torch.save({
            'model_state_dict': self.state_dict(),
        }, f"checkpoints/encoder/{file_name}.pt")


class AutoEncoderTrainer:
    def __init__(self, model, device="cpu"):
        self.evolution, _ = HaltedEvolution.defaultEvolver(steps=2000)
        self.model = model.to(device)
        self.device = device

    def cleanup(self):
        self.evolution.close()

    def save_model(self):
        self.model.save_model()

    def scoop_sample(self):
        inp, _ = self.evolution.next()
        frame = inp.astype(np.uint8)
        resized = cv2.resize(frame, dsize=(200, 200), interpolation=cv2.INTER_CUBIC)
        return resized

    def build_training_set(self):
        TARGET_TRAINING_SIZE = 10000
        TARGET_TESTING_SIZE = 2000
        TRAIN_DIR = "data/any/train/any"
        TEST_DIR = "data/any/test/any"

        curr_train_size = len(os.listdir(TRAIN_DIR))
        curr_test_size = len(os.listdir(TEST_DIR))
        remaining_training_needed = max(TARGET_TRAINING_SIZE - curr_train_size, 0)
        remaining_testing_needed = max(TARGET_TESTING_SIZE - curr_test_size, 0)

        for i in range(remaining_training_needed):
            t = round(time.time())
            matplotlib.image.imsave(f'{TRAIN_DIR}/{t}.png', self.scoop_sample(), cmap='gray')

        for i in range(remaining_testing_needed):
            t = round(time.time())
            matplotlib.image.imsave(f'{TEST_DIR}/{t}.png', self.scoop_sample(), cmap='gray')

        print("Data Generation Complete")

    def train(self):
        self.evolution.setup()
        self.build_training_set()

        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=1e-3,
                                     weight_decay=1e-8)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

        data_transform_testing = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

        training_data = torchvision.datasets.ImageFolder(root="data/any/train", transform=data_transform_testing)
        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=20,
            shuffle=True,
            num_workers=3
        )

        testing_data = torchvision.datasets.ImageFolder(root="data/any/test", transform=data_transform_testing)
        test_loader = torch.utils.data.DataLoader(
            testing_data,
            batch_size=20,
            shuffle=True,
            num_workers=3
        )

        epochs = 200

        loss_fn = torch.nn.BCELoss()

        for e in range(epochs):
            total_loss = 0.0
            for batch, (X, _) in enumerate(train_loader):
                optimizer.zero_grad()
                X = X.to(self.device)
                X = torch.reshape(X, (20, 200*200))
                decoded = self.model(X)
                loss = loss_fn(decoded, X)
                loss.backward()
                optimizer.step()
                if batch % 20 == 0:
                    print(f"Epoch: {e}, batch: {batch}")

                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)

            with torch.no_grad():
                testing_loss = 0.0
                for batch, (X, _) in enumerate(test_loader):
                    X = X.to(self.device)
                    X = torch.reshape(X, (20, 200*200))
                    decoded = self.model(X)
                    loss = loss_fn(decoded, X)
                    testing_loss += loss.item()
                testing_loss /= len(test_loader)

            print(f"Epoch: {e}, Train Loss: {avg_loss}, Test Loss: {testing_loss}")
            scheduler.step()

            if e % 2 == 0:
                with torch.no_grad():
                    rand_test = random.randint(0, len(test_loader.dataset))
                    train_img = train_loader.dataset[0][0]
                    test_img = test_loader.dataset[rand_test][0]
                    plot.figure()
                    f, axarr = plot.subplots(2, 2)

                    axarr[0][0].imshow(np.reshape(train_img, (200, 200)), cmap="Greys")
                    decoded_training = self.model(torch.reshape(train_img, (1,200*200)).to(self.device))
                    output_img = decoded_training.cpu().detach().numpy()
                    axarr[1][0].imshow(np.reshape(output_img, (200, 200)), cmap="Greys")

                    axarr[0][1].imshow(np.reshape(test_img, (200, 200)), cmap="Greys")
                    decoded_testing = self.model(torch.reshape(test_img, (1,200*200)).to(self.device))
                    output_img = decoded_testing.cpu().detach().numpy()
                    axarr[1][1].imshow(np.reshape(output_img, (200, 200)), cmap="Greys")

                    plot.show()

            if e % 15 == 0:
                self.save_model()


