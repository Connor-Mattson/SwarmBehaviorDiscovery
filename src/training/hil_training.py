import logging

import torch
import time
import os
import sys
import random
from data.swarmset import ContinuingDataset, SwarmDataset
from src.networks.embedding import NoveltyEmbedding
from src.networks.archive import DataAggregationArchive
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip
from src.networks.ensemble import Ensemble
import numpy as np
from scipy import ndimage
import random
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
import cv2

from src.networks.network_wrapper import NetworkWrapper

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def CSVLineToVec(line):
    line_list = line.strip().replace("\n", "").split(",")
    float_list = []
    for i in line_list:
        float_list.append(float(i))
    float_list = np.array(float_list)
    return float_list

def resizeInput(X, w=200):
    frame = X.astype(np.uint8)
    resized = cv2.resize(frame, dsize=(w, w), interpolation=cv2.INTER_AREA)
    return resized

def translate(img, offset=(10, 10)):
    h, w = img.shape
    xoff, yoff = offset
    if xoff < 0: xpadding = (0, -xoff)
    else: xpadding = (xoff, 0)
    if yoff < 0: ypadding = (0, -yoff)
    else: ypadding = (yoff, 0)
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
    h, w = [ zoom * i for i in img.shape ]
    if coord is None: cx, cy = w/2, h/2
    else: cx, cy = [ zoom*c for c in coord ]
    img = cv2.resize( img, (0, 0), fx=zoom, fy=zoom)
    img = img[ int(round(cy - h/zoom * .5)) : int(round(cy + h/zoom * .5)),
               int(round(cx - w/zoom * .5)) : int(round(cx + w/zoom * .5))]
    return img

def get_color_distortion(X, s=3.0):
    X = X + s * np.random.randn(X.shape[0], X.shape[1])
    return X

def getRandomFlip(X):
    tmp = torch.tensor(X).unsqueeze(0)
    flipper_A = RandomHorizontalFlip(0.5)
    flipper_B = RandomVerticalFlip(0.5)
    image = flipper_A(flipper_B(tmp))
    image = image.squeeze(0).numpy()
    return image

def getRandomTransformation(image, k=2):
    transformation_choices = ["Rotation", "Blur", "Zoom", "Translate", "Distort", "ResizedCrop"]
    # weights = [0.4, 0.3, 0.0, 0.2]
    # weights = [1.0, 0.0, 0.0, 0.0]
    # choices = random.choices(transformation_choices, weights, k=k)
    choices = ["ResizedCrop"]
    # choices = []
    if "RandomFlip" in choices:
        image = getRandomFlip(image)
    if "ResizedCrop" in choices:
        tmp = torch.tensor(image).unsqueeze(0)
        flipper = RandomHorizontalFlip(0.5)
        cropper = RandomResizedCrop(size=(50,50), scale=(0.6, 1.0), ratio=(1.0, 1.0))
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

def get_labeled_triplets(label_file):
    classes = [-1 for i in range(500)]
    num_labels = 0
    with open(label_file, "r") as f:
        lines = f.readlines()
        num_labels = len(lines)
        for i, line in enumerate(lines):
            if i > len(classes) - 1:
                break
            triplet = CSVLineToVec(line)
            classes[int(triplet[0])] = int(triplet[1])

    triplets = []

    for i, i_c in enumerate(classes):
        if i_c == 0:
            continue
        continue_to_top = False
        for j, j_c in enumerate(classes):
            if j_c != i_c:
                continue
            if i == j:
                continue
            for k, k_c in enumerate(classes):
                if k_c == 0:
                    continue
                if k_c == i_c or k_c == j_c:
                    continue
                # if i_c == 0:
                #     if not (i, i, k) in triplets:
                #         triplets.append((i, i, k))
                #         continue_to_top = True
                triplets.append((i, j, k))
            if continue_to_top:
                break
    return triplets, num_labels

def pretraining(data, network, triplets, data_cutoff=None, data_size=500, odds_original=0.6):
    if data_cutoff is None:
        data_cutoff = len(data) - 1
    random.shuffle(triplets)
    selected_labels = triplets[:data_size]
    total_loss = 0.0

    BATCH_SIZE = 4096
    total_updates = 0
    total_batches = max(len(selected_labels), data_size) // BATCH_SIZE

    # Batch the data
    for i in range(0, len(selected_labels), BATCH_SIZE):
        # AUGMENT_SIZE = 1
        if i + BATCH_SIZE >= len(selected_labels):
            continue

        print(f"Human in the Loop Training.. {(total_updates * 100) / total_batches}")

        anchors = np.array([data[selected_labels[i + j][0]][0] for j in range(BATCH_SIZE)])

        train_original = random.random() < odds_original
        if train_original:
            positives = np.array(
                [
                    getRandomTransformation(data[selected_labels[i + j][0]][0]) for j in range(BATCH_SIZE)
                ]
            )
        else:
            positives = np.array(
                [
                    getRandomFlip(data[selected_labels[i + j][1]][0]) for j in range(BATCH_SIZE)
                ]
            )

        negatives = np.array([data[selected_labels[i + j][2]][0] for j in range(BATCH_SIZE)])

        anchors = np.expand_dims(anchors, axis=1)
        positives = np.expand_dims(positives, axis=1)
        negatives = np.expand_dims(negatives, axis=1)

        loss = network.train_batch(anchors, positives, negatives)
        total_loss += loss
        total_updates += 1

    return total_loss, max(total_updates, 1)

def human_label_training(output_dir, export_name, dataset, checkpoint, labels, lr=3e-2, epochs=500, seed=None):
    logging.info("Beginning Human in the Loop Training From Label Set")
    logging.info(f"Creating triplets from human labels {labels}")
    triplets, n = get_labeled_triplets(labels)
    logging.info(f"File retrieved with {n} labels, forming {len(triplets)} triplets")

    TOTAL_EPOCHS = epochs
    target = 0.0001
    loss = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Training model on device: {device}")

    network = NetworkWrapper(output_size=5, lr=lr, margin=1.0, weight_decay=1e-6, new_model=True, manual_schedulers=True, dynamic_lr=True, warmup=5)
    if checkpoint is not None:
        network.load_from_path(checkpoint)

    sampled_dataset = SwarmDataset(dataset, rank=0)
    start_time = time.time()

    # Training Loop
    e = 0
    loss_history = []
    while e < TOTAL_EPOCHS:
        total_loss, total_updates = pretraining(sampled_dataset, network, triplets, data_size=5000, odds_original=0.05)
        curr_loss = total_loss / total_updates
        loss_history.append(curr_loss)
        average_loss = (sum(loss_history[-3:]) / 3) if len(loss_history) > 3 else 50
        logging.info(f"Epoch {e}, loss: {curr_loss}, Average Window Loss: {average_loss}, LR: {network.get_lr()}")
        e += 1
        network.step_scheduler(loss)

        if average_loss < target:
            break

    logging.info(f"Total HIL Training Time: {time.time() - start_time}")

    # Save Model
    network.save(output_dir, export_name)



