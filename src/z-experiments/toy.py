import torch
import time
from data.swarmset import ContinuingDataset, SwarmDataset
from src.networks.embedding import NoveltyEmbedding
from src.networks.archive import DataAggregationArchive
from src.networks.ensemble import Ensemble
import numpy as np
from scipy import ndimage
import random
classification_set = {
    0 : [],
    1 : [],
    2 : []
}

PRETRAINING = True
target = 0.01
loss = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ensemble = Ensemble(size=3, output_size=5, lr_series=[15e-4, 15e-4, 15e-4], learning_decay=0.9, decay_step=4, threshold=6.0, weight_decay=1e-4)
ensemble.load_ensemble("toy-surgery")
sampled_dataset = SwarmDataset("data/toy", rank=0)
data = sampled_dataset

# Separate
for i in range(len(sampled_dataset)):
    _class = sampled_dataset[i][1][0]
    classification_set[_class].append(i)

# Pair Up
SAMPLES = 3000
triplets = []
for i in range(SAMPLES):
    classA = random.randint(0, 2)
    classB = random.randint(0, 2)

    anchor = random.choice(classification_set[classA])
    positive = random.choice(classification_set[classA])
    negative = random.choice(classification_set[classB])
    triplet = [anchor, positive, negative]
    if triplet not in triplets:
        triplets.append(triplet)

# Training
BATCH_SIZE = 4096
EPOCH_DATA_LIM = 500
while loss > target:
    total_updates = 0
    total_loss = np.array([0.0 for i in range(len(ensemble.ensemble))])
    random.shuffle(triplets)
    temp_triplets = triplets[:EPOCH_DATA_LIM]
    for i in range(0, len(temp_triplets), BATCH_SIZE):
        if total_updates % 10 == 0:
            print(f"Unsupervised Training.. {(total_updates * BATCH_SIZE * 100) / len(temp_triplets)}")

        if i + BATCH_SIZE > len(triplets):
            break

        anchors = np.array([data[temp_triplets[i + j][0]][0] for j in range(BATCH_SIZE)])
        positives = np.array([data[temp_triplets[i + j][1]][0] for j in range(BATCH_SIZE)])
        negatives = np.array([data[temp_triplets[i + j][2]][0] for j in range(BATCH_SIZE)])

        anchors = np.expand_dims(anchors, axis=1)
        positives = np.expand_dims(positives, axis=1)
        negatives = np.expand_dims(negatives, axis=1)

        losses = ensemble.train_batch(anchors, positives, negatives)
        total_loss += losses
        total_updates += 1

    l = total_loss / total_updates
    lr = ensemble.evaluate_lr(l)
    loss = sum(l) / len(l)
    print(f"Losses: {l}, LR: {lr}, Loss: {loss}")

print("Complete!")
# ensemble.save_ensemble(f"{int(time.time())}")


