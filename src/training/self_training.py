import torch
import time
import random
from absl import logging
from data.swarmset import ContinuingDataset, SwarmDataset
from src.networks.embedding import NoveltyEmbedding
from src.networks.archive import DataAggregationArchive
from src.networks.network_wrapper import NetworkWrapper
import numpy as np
from scipy import ndimage
import cv2
from torchlars import LARS
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip


def resizeInput(X, w=200):
    frame = X.astype(np.uint8)
    resized = cv2.resize(frame, dsize=(w, w), interpolation=cv2.INTER_AREA)
    return resized


def translate(img, offset=(10, 10)):
    h, w = img.shape
    xoff, yoff = offset
    if xoff < 0:
        xpadding = (0, -xoff)
    else:
        xpadding = (xoff, 0)
    if yoff < 0:
        ypadding = (0, -yoff)
    else:
        ypadding = (yoff, 0)
    img = np.pad(img, (xpadding, ypadding))

    if xoff >= 0 and yoff >= 0:
        return img[:w, :w]
    elif xoff < 0 and yoff >= 0:
        return img[-w:, :w]
    elif xoff >= 0 and yoff < 0:
        return img[:w, -w:]
    return img[-w:, -w:]


def zoom_at(img, zoom, coord=None):
    # Adapted from https://stackoverflow.com/questions/69050464/zoom-into-image-with-opencv
    h, w = [zoom * i for i in img.shape]
    if coord is None:
        cx, cy = w / 2, h / 2
    else:
        cx, cy = [zoom * c for c in coord]
    img = cv2.resize(img, (0, 0), fx=zoom, fy=zoom)
    img = img[int(round(cy - h / zoom * .5)): int(round(cy + h / zoom * .5)),
          int(round(cx - w / zoom * .5)): int(round(cx + w / zoom * .5))]
    return img


def get_color_distortion(X, s=3.0):
    X = X + s * np.random.randn(X.shape[0], X.shape[1])
    return X


def getRandomTransformation(image, k=2):
    transformation_choices = ["Rotation", "Blur", "Zoom", "Translate", "Distort", "ResizedCrop"]
    # weights = [0.4, 0.3, 0.0, 0.2]
    # weights = [1.0, 0.0, 0.0, 0.0]
    # choices = random.choices(transformation_choices, weights, k=k)
    choices = ["ResizedCrop", "Rotation"]
    if "ResizedCrop" in choices:
        tmp = torch.tensor(image).unsqueeze(0)
        flipper = RandomHorizontalFlip(0.5)
        cropper = RandomResizedCrop(size=(50, 50), scale=(0.6, 1.0), ratio=(1.0, 1.0))
        image = flipper(cropper(tmp))
        image = image.squeeze(0).numpy()
    if "Rotation" in choices:
        theta = random.choice([90, 180, 270])
        image = ndimage.rotate(image, theta)
    if "Blur" in choices:
        blur = random.choice([0.5, 1.0, 1.5])
        image = ndimage.gaussian_filter(image, sigma=blur)
    if "Zoom" in choices:
        # zoom = random.choice([1.06, 1.12, 1.18])
        padding = random.choice([10])
        padded = np.pad(image, padding, mode='constant')
        image = resizeInput(padded, 50)
    if "Translate" in choices:
        offsets = [i for i in range(-10, 10, 2)]
        offset = (random.choice(offsets), random.choice(offsets))
        # offset = (2, 2)
        image = translate(image, offset)
    if "Distort" in choices:
        strength = random.choice([3.0, 5.0, 10.0])
        image = get_color_distortion(image, s=strength)
    if "Flip" in choices:
        tmp = torch.tensor(image).unsqueeze(0)
        flipper = RandomHorizontalFlip(1.0)
        image = flipper(tmp)
        image = image.squeeze(0).numpy()
    return image


def triplet_negative_mining(anchor_embeddings, positive_embeddings, m):
    negatives = []
    for i, a in enumerate(anchor_embeddings):
        d_p = torch.linalg.norm(a - positive_embeddings[i])
        found = False
        for j, n in enumerate(anchor_embeddings):
            d_n = torch.linalg.norm(n - a)
            if d_p < d_n < d_p + m:  # Semi-Hard Triplet
                # print(f"Semi-Hard Found! P: {d_p}, N: {d_n}")
                negatives.append(j)
                found = True
                break
        if not found:
            negatives.append(random.randint(0, len(anchor_embeddings) - 1))
    return negatives

def pretraining(data, network, data_cutoff=None, update_count=6):
    if data_cutoff is None:
        data_cutoff = len(data) - 1

    # BATCH_SIZE = 4096
    BATCH_SIZE = 2048
    samples = np.random.random_integers(0, data_cutoff, (BATCH_SIZE * update_count))
    total_loss = 0.0
    total_updates = 0

    # Batch the data
    for i in range(0, len(samples), BATCH_SIZE):
        # AUGMENT_SIZE = 1
        # if i + (BATCH_SIZE * AUGMENT_SIZE) >= len(pull_set):
        #     continue

        # print(f"Unsupervised Training.. {(total_updates * 100) / total_batches}")
        anchors = np.array([data[samples[i + j]][0] for j in range(BATCH_SIZE)])
        positives = np.array([getRandomTransformation(data[samples[i + j]][0]) for j in range(BATCH_SIZE)])
        anchors_expanded = np.expand_dims(anchors, axis=1)
        positives = np.expand_dims(positives, axis=1)

        anchor_embeddings = network.batch_out(anchors_expanded)
        positive_embeddings = network.batch_out(positives)
        negatives_indices = triplet_negative_mining(anchor_embeddings, positive_embeddings, network.margin)

        negatives = np.array([anchors[j] for j in negatives_indices])
        negatives = np.expand_dims(negatives, axis=1)

        loss = network.train_batch(anchors_expanded, positives, negatives)
        total_loss += loss
        total_updates += 1

    return total_loss, max(total_updates, 1)


def self_supervised_training(output_dir, export_name, dataset, lr=3e-2, epochs=500, seed=None):
    if seed is not None:
        GLOBAL_SEED = seed
        np.random.seed(GLOBAL_SEED)
        random.seed(GLOBAL_SEED)
        torch.manual_seed(GLOBAL_SEED)

    TOTAL_EPOCHS = epochs
    target = 0.0001
    loss = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Training model on device: {device}")

    network = NetworkWrapper(output_size=5, lr=lr, margin=1.0, weight_decay=1e-6, new_model=True,
                             manual_schedulers=True, dynamic_lr=True, warmup=5)
    sampled_dataset = SwarmDataset(dataset, rank=0)

    start_time = time.time()

    # Training Loop
    e = 0
    loss_history = []
    while e < TOTAL_EPOCHS:
        total_loss, total_updates = pretraining(sampled_dataset, network, update_count=1)
        curr_loss = total_loss / total_updates
        loss_history.append(curr_loss)
        average_loss = (sum(loss_history[-3:]) / 3) if len(loss_history) > 3 else 50
        logging.info(f"Epoch {e}, loss: {curr_loss}, Average Window Loss: {average_loss}, LR: {network.get_lr()}")
        e += 1
        network.step_scheduler(loss)

        if average_loss < target:
            break

    logging.info(f"Total Self-Supervised Training Time: {time.time() - start_time}")

    # Save Model
    network.save(output_dir, export_name)
