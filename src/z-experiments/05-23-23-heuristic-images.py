"""
The purpose of this experiment is to determine whether it is possible to distinguish between coherent
and entropic behaviors using image data. This could act as another heuristic filter in addition
to the hand-crafted filters outlined in the GECCO paper.
First, we'll try analyzing the images using statistics to see if we can pick out a pattern.
If that doesn't work, we'll try training a CNN model instead.

Author: Jeremy Clark
"""

import random
import multiprocessing as mp
import pygame
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from novel_swarms.world.RectangularWorld import RectangularWorld
from data.img_process import get_image_map, generate_world_config

# How many images should we collect for training
sample_size = 1000

# Which portions of the program we should execute
image_collection = True
labeling = False
stats = False


def _generate_random_controllers(num=6, length=4):
    """
    Generate random controllers.
    Each random controller is normalized to the highest absolute value in that controller.

    :param num: the number of controllers to generate
    :param length: how long each controller should be
    :return: a list of normalized random controllers
    """
    controls = []
    for _ in range(num):
        # All elements in the controller should be randomly selected between -1 and 1
        control = [random.uniform(-1, 1) for _ in range(length)]
        magnitudes = [abs(x) for x in control]
        scaling_factor = 1 / max(magnitudes)
        normalized_controller = [scaling_factor * x for x in control]
        controls.append(normalized_controller)
    return controls


def _get_both_maps(control, index):
    """
    Saves the image representations of the given controller as both a density map and a trail map.
    Useful for multiprocessing because you only have to step through the world 1500 times once overall,
    instead of stepping through the world 1500 times for each image representation.

    :param control: the controller for which we are getting the image maps for
    :param index: the index of the controller in controller_list
    :return: None, but save the trail-map and density-map representations of this controller as images
    """
    # Update the user about the progress of the data collection process
    if index % 10 == 0:
        progress_percent = round(100 * (index / sample_size), 2)
        print(f"Progress: {progress_percent}%")

    # Create a shared world and step through 1500 times
    config = generate_world_config(control)
    world = RectangularWorld(config)
    for _ in range(1500):
        world.step()
    # Save the density-map and trail-map in their appropriate locations
    trail_path = f"../../data/trail-maps/controller-{index}.jpg"
    density_path = f"../../data/density-maps/controller-{index}.jpg"
    get_image_map(control, "trail", filepath=trail_path, world=world)
    get_image_map(control, "density", filepath=density_path, world=world)


def label_controller(index):
    """
    Returns whether the controller at the given index produces a coherent behavior (1) or not (0)
    Gets the image representation of the controller corresponding to the given index and displays it to the user.
    If the user presses 0, that indicates that the controller produces an entropic behavior. If they press
    1, then it means that the controller produces a coherent behavior.

    :param index: the index corresponding to a controller in controller_list
    :return: int representing the coherence of the controller at the given index
    """
    trail_path = f"../../data/trail-maps/controller-{index}.jpg"
    density_path = f"../../data/density-maps/controller-{index}.jpg"
    trail_map_image = None
    density_map_image = None
    # Try to get the images if they exist. If they don't return early
    try:
        trail_map_image = pygame.image.load(trail_path)
        density_map_image = pygame.image.load(density_path)
    except FileNotFoundError:
        return "File not found"

    # Start pygame
    pygame.init()
    screen = pygame.display.set_mode((620, 500))

    coherent = None  # We don't know whether the behavior is coherent
    font = pygame.font.Font(None, 25)

    running = True
    while running:
        for event in pygame.event.get():
            # If the user tries to close the window, quit
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                # If the user presses 0, stop running and label this index as entropic
                if event.key == pygame.K_0:
                    coherent = 0
                    running = False
                # If the user presses 1, stop running and label this index as coherent
                elif event.key == pygame.K_1:
                    coherent = 1
                    running = False

        screen.fill((0, 0, 0))  # Fill with black
        screen.blit(trail_map_image, (0, 0))  # Put the image on the screen
        screen.blit(density_map_image, (500, 70))

        # Provide a reference to the user
        text = font.render("0 = entropic", True, (200, 200, 200))
        screen.blit(text, (505, 5))
        text = font.render("1 = coherent", True, (200, 200, 200))
        screen.blit(text, (505, 30))
        text = font.render(f"index = {index}", True, (200, 200, 200))
        screen.blit(text, (505, 475))

        # Refresh the screen and wait for 1/5 of a second
        pygame.display.flip()
        pygame.time.delay(200)

    return coherent


controller_list = _generate_random_controllers(num=sample_size)  # Generate a dataset of random controllers

# If the user has enabled image collection, collect and store image + controller data using multiprocessing
csv_filepath = "../../data/labeled-controllers.csv"
if image_collection:
    df = pd.DataFrame(controller_list)
    df.insert(0, None, 0)
    df.to_csv(csv_filepath, index=False, header=False)

    max_processes = 4
    pool = mp.Pool(processes=max_processes)
    for i, c in enumerate(controller_list):
        pool.apply_async(_get_both_maps, args=(c, i))
    pool.close()
    pool.join()

    # for i, c in enumerate(controller_list):
    #     _get_both_maps(c, i)

# If the user has enabled labeling, loop over the stored controllers and allow the user to label them
if labeling:
    df = pd.read_csv(csv_filepath, header=None)
    for i in range(len(controller_list)):
        if (i+1) % 50 == 0:
            df.to_csv(csv_filepath, index=False, header=False)

        label = label_controller(i)
        # Handle cases where the file doesn't exist
        if not label == "File not found":
            df.loc[i, 0] = label
        else:
            print(f"Faulty index: {i}")

    df.to_csv(csv_filepath, index=False, header=False)

if stats:
    labeled_controllers = pd.read_csv(csv_filepath, header=None).to_numpy()
    mean_light_values = []
    labels = []
    for i, row in enumerate(labeled_controllers):
        label = row[0]
        labels.append(label)
        controller = row[1:].tolist()
        density_filepath = f"../../data/trail-maps/controller-{i}.jpg"
        density_image = None

        try:
            img = Image.open(density_filepath).convert('L')
            image_array = np.array(img)
            image_array_normalized = (image_array / np.max(image_array)) * 255
            density_image = image_array_normalized.astype(np.uint8)
        except FileNotFoundError:
            mean_light_values.append(None)
            continue

        mean_light_values.append(np.mean(density_image))

    entropic_light_values = []
    coherent_light_values = []
    for i in range(len(mean_light_values)):
        if mean_light_values[i] is None:
            continue

        if labels[i] == 0:
            entropic_light_values.append(mean_light_values[i])
        elif labels[i] == 1:
            coherent_light_values.append(mean_light_values[i])

    # Plotting the histogram
    plt.hist(coherent_light_values, bins=range(160, 256, 7), color='green', alpha=0.5, label='Coherent')
    plt.hist(entropic_light_values, bins=range(160, 256, 7), color='red', alpha=0.5, label='Entropic')

    # Chart settings
    plt.title('Average Brightness Histogram')
    plt.xlabel('Brightness Value')
    plt.ylabel('Frequency')
    plt.legend()

    # Saving the chart as a PNG
    my_filepath = 'path_to_save_image.png'
    plt.savefig(my_filepath)

    # Display the chart
    plt.show()


print("Done.")
