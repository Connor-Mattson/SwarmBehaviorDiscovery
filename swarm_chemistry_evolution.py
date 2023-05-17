"""
Implements the technique shown in Swarm Chemistry:
https://direct.mit.edu/artl/article-abstract/15/1/105/2623/Swarm-Chemistry

Human picks 2 out of 8 behaviors shown on a GUI.
These two controllers are combined using the novel_swarms.BehaviorDiscovery.crossOver() function.
"""
import math

import random
import numpy as np
import pygame
from PIL import Image
import multiprocessing

# These imports are required for world creation
from novel_swarms.config.AgentConfig import DiffDriveAgentConfig
from novel_swarms.config.WorldConfig import RectangularWorldConfig
from novel_swarms.gui.agentGUI import DifferentialDriveGUI
from novel_swarms.sensors.BinaryLOSSensor import BinaryLOSSensor
from novel_swarms.sensors.SensorSet import SensorSet
from novel_swarms.world.RectangularWorld import RectangularWorld
from novel_swarms.util.timer import Timer
from scipy.ndimage import gaussian_filter
from novel_swarms.world.simulate import main as simulate

"""
Helper function for evaluate_density_map
Given a matrix indicating the number of agents at each point, along with an index of that matrix,
returns the sum of all of the agents within a 100x100 box centered at that index.
Overflow is accounted for in evaluate_denisty_map
"""
def _scan_kernel(point_map, h, w):
    # Get window area
    row_start = h - 50
    row_end = h + 50
    col_start = w - 50
    col_end = w + 50
    kernel_size = 101

    sigma = 20  # Standard deviation of gaussian blur
    kernel = np.zeros((kernel_size, kernel_size))  # Create a 100x100 kernel
    kernel[kernel_size // 2, kernel_size // 2] = 1  # Set center of kernel window to 1
    kernel = gaussian_filter(kernel, sigma=sigma)  # Apply gaussian blur out from center
    window = point_map[row_start:row_end + 1, col_start:col_end + 1] * kernel
    return np.sum(window)


"""
Helper function for evaluate_density_map
Normalizes the density map so that all of the values in the map are between 0 and 1.
"""


def _normalize_density_map(density_map):
    return density_map / np.max(density_map)


"""
IMPORTANT: In order for this function to work, density_map_on MUST be set to True in world/simulate.py
This is an alternative version of evaluate that outputs a density map of the agents over a specified
number of steps. Skip specifies the frequency at which data should be collected.
For example, if step = 50 and skip = 10, then the density map will contain the positions of agents at
steps 0, 10, 20, 30, 40

Padding = 50
Stride = 5
Kernel = 100
"""
def _evaluate_density_map(world, steps: int, filepath, skip=10, stride=5):
    # Keeps track of the number of agents at each point. Padded by 50 on all sides
    point_map = np.zeros((world.bounded_height + 100, world.bounded_width + 100), dtype=int)

    # Populate the point map by stepping, getting the positions of agents at each step, and adding them to the map
    for step in range(steps * skip):
        world.step()
        if step % skip == 0:  # If we should collect data about the agents' positions
            for agent in world.population:
                x = math.floor(agent.getPosition()[0] + 50)
                y = math.floor(agent.getPosition()[1] + 50)
                point_map[y, x] += 1
    # Create density map with all density starting at 0
    # Dimensions of the map will be significantly smaller than the dimensions of the arena because stride = 5
    output_shape = (math.floor(world.bounded_height / stride), math.floor(world.bounded_width / stride))
    output = np.zeros(output_shape, dtype=float)
    output_h = 0
    output_w = 0
    # Iterate over the point map using a stride of 5
    for h in range(50, world.bounded_height + 50, stride):
        for w in range(50, world.bounded_width + 50, stride):
            # The value at output[output_h, output_w] represents the density of agents at that point
            output[output_w, output_h] = _scan_kernel(world, point_map, h, w)
            output_w += 1
        output_h += 1
        output_w = 0
    # Ensures that all values in the map are between 0 and 1 so that we can represent it as a single-channel image

    imgarray = _normalize_density_map(world, output) * 255
    img = Image.fromarray(imgarray.astype('uint8'), 'L')
    img.save(filepath)

"""
Returns a square 2d numpy array with dimensions twice the given radius.
For example, if the radius is 5, the dimensions will be 10x10.
Elements that are more than `radius` elements away from the center of the window are ones
and the rest are zeroes. This results in a circular mask.
This is used to show where agents have been on the map. The radius should
correspond to the radius of the agents, which is why radius is set to 5.
"""
def _make_circle_mask(radius=5):
    mask = np.zeros((radius * 2, radius * 2), dtype=int)
    for (h, w), _ in np.ndenumerate(mask):
        y = (h + 1) - 5
        x = (w + 1) - 5
        dist = math.sqrt(math.pow(y, 2) + math.pow(x, 2))
        if dist > 5:
            mask[h, w] = 1
    return mask


"""
Applies circular mask to an area of the output image.
"""
def _add_circle_mask(output, x, y, mask):
    region = output[y:y + 10, x:x + 10]
    output[y:y + 10, x:x + 10] *= mask
    return output


"""
Applies circular masks to the output that correspond to the positions of
every agent in the world.
"""
def _collect_data(world, output):
    circle_mask = _make_circle_mask()
    new_output = output
    for agent in world.population:
        agent_position = agent.getPosition()
        x = int(agent_position[0])
        y = int(agent_position[1])
        new_output = _add_circle_mask(output, x - 5, y - 5, circle_mask)
    return new_output


"""
Scans the simulation over a specified number of timesteps and saves an image
of where the agents have been over those timesteps in a specified location
"""
def _evaluate_trails(world, steps: int, filepath, skip=3):
    output = np.ones((500, 500), dtype=int)
    for i in range(steps * skip):
        world.step()
        if i % skip == 0:
            output = _collect_data(world, output)
    output *= 255
    img = Image.fromarray(output.astype('uint8'), 'L')
    img.save(filepath)


def invisible_simulate(world_config, img_rep, filepath):
    total_allowed_steps = 1200

    density_map_on = False
    trail_map_on = False
    if img_rep == "density-map":
        density_map_on = True
    if img_rep == "trail-map":
        trail_map_on = True

    # Create the simulation world
    world = RectangularWorld(world_config)

    steps_taken = 0
    steps_per_frame = 1

    running = True
    while running:
        # Calculate Steps - Stop if we reach desired frame
        for _ in range(steps_per_frame):
            if total_allowed_steps is not None:
                if steps_taken > total_allowed_steps:
                    running = False
                    if density_map_on:
                        _evaluate_density_map(world, 500, filepath)

                    if trail_map_on:
                        _evaluate_trails(world, 20, filepath)

                    break

            world.step()
            steps_taken += 1

def _generate_world_config(controller):
    behavior = []
    sensors = SensorSet([BinaryLOSSensor(angle=0)])
    agent_config = DiffDriveAgentConfig(
        controller=controller,
        sensors=sensors,
        seed=None,
    )
    config = RectangularWorldConfig(
        size=(500, 500),
        n_agents=30,
        seed=None,
        behavior=behavior,
        agentConfig=agent_config,
        padding=15,
        show_walls=True,
        collide_walls=True
    )
    return config

def get_trail_map(controller, filepath):
    config = _generate_world_config(controller)
    invisible_simulate(config, "trail-map", filepath)

def _generate_random_controllers(num=6):
    controller_list = []
    for _ in range(num):
        controller = [random.uniform(0, 1) for _ in range(4)]
        controller_list.append(controller)
    return controller_list

"""
Returns index of the tile
"""
def _calculate_tile_clicked(mouse_pos):
    x, y = mouse_pos
    tile_x = x // 500
    tile_y = y // 500
    tile_pos = (tile_x, tile_y)
    pos_dict = {
        (0, 0): 0,
        (1, 0): 1,
        (2, 0): 2,
        (0, 1): 3,
        (1, 1): 4,
        (2, 1): 5
    }
    if tile_pos not in pos_dict.keys():
        print("User has clicked outside of the window")
        return
    return pos_dict[tile_pos]

def HIL_evolution_GUI(controllers):
    pygame.init()

    timesteps = 1200
    gui_width = 1500
    gui_height = 1000
    tile_height = 500
    tile_width = 500
    parent_screen = pygame.display.set_mode((gui_width, gui_height))
    worlds = [RectangularWorld(_generate_world_config(e)) for e in controllers]

    # def async_step(w):
    #     for _ in range(timesteps):
    #         w.step()
    #
    # def step_concurrently():
    #     processes = []
    #     for w in worlds:
    #         p = multiprocessing.Process(target=async_step, args=(w,))
    #         processes.append(p)
    #         p.start()
    #
    #     for p in processes:
    #         p.join()
    #
    # step_concurrently()
    # for world in worlds:
    #
    filter_surface = pygame.Surface((tile_width, tile_height), pygame.SRCALPHA)
    filter_surface.fill((255, 255, 255, 50))

    tiles = []
    tile_positions = [
        (0, 0),
        (500, 0),
        (1000, 0),
        (0, 500),
        (500, 500),
        (1000, 500)
    ]
    for _ in range(6):
        tile = pygame.Surface((tile_width, tile_height))
        tiles.append(tile)

    running = True

    highlighted_indices = set()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # If the left mouse button has just been lifted
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                tile_index = _calculate_tile_clicked(pygame.mouse.get_pos())
                if tile_index in highlighted_indices:
                    highlighted_indices.remove(tile_index)
                else:
                    highlighted_indices.add(tile_index)
                if len(highlighted_indices) > 2:
                    highlighted_indices = set()
                print(highlighted_indices)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                if len(highlighted_indices) == 2:
                    l = list(highlighted_indices)
                    i1 = l[0]
                    i2 = l[1]
                    selected_controllers = (controllers[i1], controllers[i2])
                    print(selected_controllers)
                    return selected_controllers

        parent_screen.fill((0, 0, 0))

        for i in range(6):
            current_tile = tiles[i]
            tile_world = worlds[i]
            tile_world.step()
            current_tile.fill((0, 0, 0))
            tile_world.draw(current_tile)
            parent_screen.blit(current_tile, tile_positions[i])
            if i in highlighted_indices:
                parent_screen.blit(filter_surface, tile_positions[i])

        pygame.display.flip()
        pygame.time.Clock().tick(500)


if __name__ == '__main__':
    HIL_evolution_GUI(_generate_random_controllers())







