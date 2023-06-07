"""
Image processing package for swarm behavior.
This package contains two functions, `get-trail-map` and `get-density-map`,
which generate and save image representations of a given controller.

Author: Jeremy Clark
"""
import math
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import pygame
import imageio
import tempfile
import os
from functools import lru_cache

from novel_swarms.config.AgentConfig import DiffDriveAgentConfig
from novel_swarms.config.WorldConfig import RectangularWorldConfig
from novel_swarms.sensors.BinaryLOSSensor import BinaryLOSSensor
from novel_swarms.sensors.SensorSet import SensorSet
from novel_swarms.world.RectangularWorld import RectangularWorld


@lru_cache(maxsize=5)
def _make_circle_mask(radius=5):
    """
    Returns a square 2d numpy array with dimensions twice the given radius.
    For example, if the radius is 5, the dimensions will be 10x10.
    Elements that are more than `radius` elements away from the center of the window are ones
    and the rest are zeroes. This results in a circular mask.
    This is used to show where agents have been on the map. The radius should
    correspond to the radius of the agents, which is why radius is set to 5.

    :param radius: Radius of the circle in pixels. Should correspond to the radii of the agents.
    :return: a circular mask
    """
    mask = np.zeros((radius * 2, radius * 2), dtype=int)
    for (h, w), _ in np.ndenumerate(mask):
        y = (h + 1) - 5
        x = (w + 1) - 5
        dist = math.sqrt(math.pow(y, 2) + math.pow(x, 2))
        if dist > 5:
            mask[h, w] = 1
    return mask


def _generate_blur_kernel(dim, sigma):
    """
    Generates a blur kernel that will be passed as a parameter to _scan_kernel so that it doesn't have
    to compute the same kernel every single time.

    :param dim: the height and width (pixels) of the square kernel
    :param sigma: the standard deviation of the blur
    :return: a gaussian blur kernel
    """
    if dim % 2 == 0:
        dim += 1
    kernel = np.zeros((dim, dim))  # Create a kernel of the specified dimensions
    kernel[dim // 2, dim // 2] = 1  # Set center of kernel window to 1
    kernel = gaussian_filter(kernel, sigma=sigma)  # Apply gaussian blur out from center
    return kernel


def _scan_kernel(point_map, kernel, dim, h, w):
    """
    Helper function for _evaluate_density_map
    Given a matrix indicating the number of agents at each point, along with an index of that matrix,
    returns the sum of all the agents within a 100x100 box centered at that index.
    Overflow is accounted for in _evaluate_density_map

    :param point_map: A matrix indicating the number of agents at each point
    :param h: y-coordinate of the point around which we are scanning
    :param w: x-coordinate of the point around which we are scanning
    :return: The sum of all the agents within a 100x100 box (kernel) centered at (h, w)
    """
    # Get window area
    pad = dim // 2
    row_start = h - pad
    row_end = h + pad
    col_start = w - pad
    col_end = w + pad

    window = point_map[row_start:row_end + 1, col_start:col_end + 1] * kernel
    return np.sum(window)


def _evaluate_density_map(world, steps=2000, skip=10, stride=5, kernel_dim=100, sigma=20):
    """
    This is an alternative version of evaluate that outputs a density map of the agents over a specified
    number of steps. Skip specifies the frequency at which data should be collected.
    It's basically a homemade version of a convolution.
    For example, if step = 50 and skip = 10, then the density map will contain the positions of agents at
    steps 0, 10, 20, 30, 40

    :param world: The world that we're evaluating
    :param steps: How many steps to take
    :param skip: How often we should collect positional data on the agents
    :param stride: The stride of the image convolution
    :param kernel_dim: The dimension of the square kernel in pixels
    :return:
    """
    if kernel_dim % 2 == 1:
        kernel_dim -= 1

    pad = kernel_dim // 2
    # Keeps track of the number of agents at each point. Padded by 50 on all sides
    point_map = np.zeros((world.bounded_height + kernel_dim, world.bounded_width + kernel_dim), dtype=int)

    # Populate the point map by stepping, getting the positions of agents at each step, and adding them to the map
    for step in range(steps):
        world.step()
        if step % skip == 0:  # If we should collect data about the agents' positions
            for agent in world.population:
                x = math.floor(agent.getPosition()[0] + pad)
                y = math.floor(agent.getPosition()[1] + pad)
                point_map[y, x] += 1
    # Create density map with all density starting at 0
    # Dimensions of the map will be significantly smaller than the dimensions of the arena because stride = 5
    output_shape = (math.floor(world.bounded_height / stride), math.floor(world.bounded_width / stride))
    output = np.zeros(output_shape, dtype=float)
    output_h = 0
    output_w = 0
    # Pre-generate the blur kernel to pass it as a parameter to _scan_kernel
    kernel = _generate_blur_kernel(dim=kernel_dim, sigma=sigma)
    # Iterate over the point map using a stride of 5
    for h in range(pad, world.bounded_height + pad, stride):
        for w in range(pad, world.bounded_width + pad, stride):
            # The value at output[output_h, output_w] represents the density of agents at that point
            output[output_w, output_h] = _scan_kernel(point_map, kernel, kernel_dim, h, w)
            output_w += 1
        output_h += 1
        output_w = 0
    # Ensures that all values in the map are between 0 and 1 so that we can represent it as a single-channel image

    # Return the normalized output
    return (output / np.max(output)) * 255


def _apply_circle_mask(output, x, y, radius):
    """
    Applies circular mask to an area of the output image.
    Mutates output outside of this lexical scope.
    """
    x -= radius
    y -= radius
    # For some reason this very rarely throws an exception, in which case just skip it
    try:
        output[y:y+radius*2, x:x+radius*2] *= _make_circle_mask()
    except ValueError:
        pass


def _collect_trail_data(world, output, radius=5):
    """
    Helper function for collecting trail images.
    Applies circular masks to the output that correspond to the positions of
    every agent in the world. Does this for only one timestep.
    """
    for agent in world.population:
        agent_position = agent.getPosition()
        x = int(agent_position[0])
        y = int(agent_position[1])
        _apply_circle_mask(output, x, y, radius)


def _evaluate_trails(world, world_dims=(500, 500), steps=100, skip=3):
    """
    Scans the simulation over a specified number of timesteps and saves an image
    of where the agents have been over those timesteps in a specified location

    :param world: The world at the timestep that we want to evaluate it at
    :param steps: How many steps to take
    :param skip: How many steps between data collection
    :return: The image of the agent trails over a specified number of timesteps
    """
    output = np.ones(world_dims, dtype=int)
    for i in range(steps * skip):
        world.step()
        if i % skip == 0:
            _collect_trail_data(world, output)
    output *= 255
    return output


def generate_world_config(controller):
    """
    Generates a world configuration based on the given controller
    """
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
        behavior=[],
        agentConfig=agent_config,
        padding=15,
        show_walls=True,
        collide_walls=True
    )
    return config


def get_gif_representation(world, filepath, steps=500, skip=5):
    """
    Given a world and a filepath, saves the world as a GIF over `steps` timesteps
    in the filepath. `skip` specifies the number of steps between data collection.
    """
    screen = pygame.display.set_mode((500, 500), flags=pygame.HIDDEN)
    # A temporary directory to store all the frames of the GIF
    temp_dir = tempfile.TemporaryDirectory()

    for i in range(steps):
        world.step()
        screen.fill((0, 0, 0))
        # If it's time to collect image data, save it to the temporary directory
        if i % skip == 0:
            world.draw(screen)
            pygame.display.flip()
            frame_number = i // skip
            frame = pygame.surfarray.array3d(screen)
            img = Image.fromarray(frame.astype('uint8'), 'RGB')
            temp_path = os.path.join(temp_dir.name, f"frame-{frame_number}.jpg")
            img.save(temp_path)

    number_of_frames = steps // skip  # The number of frames in the GIF
    frames = []
    # Iterate over all the frames and combine them into a single GIF
    for i in range(number_of_frames):
        img = imageio.v2.imread(os.path.join(temp_dir.name, f"frame-{i}.jpg"))
        frames.append(img)
    imageio.mimsave(filepath, frames, duration=100, loop=0)
    # Delete the contents of the temporary directory
    temp_dir.cleanup()


def get_image_map(controller, representation, filepath=None, frame_start=1200, world=None):
    """
    Driver function for image representations. Gets the given type of representation for the given controller.

    :param controller: The controller for the swarm
    :param representation: Which function to use to analyze the image
    :param filepath: Where to save the map
    :param frame_start: Where to start recording data if a world is not provided.
    :param world: The given world. If None, generate it from scratch.
    """
    # Dictionary the maps the given `representation` to the appropriate function
    func_dict = {
        "density": _evaluate_density_map,
        "trail": _evaluate_trails,
        "gif": None
    }
    # The function used to analyze the image
    analyzer = func_dict[representation]

    # If a world was not provided, create one and step it to `frame_start`
    if world is None:
        config = generate_world_config(controller)
        world = RectangularWorld(config)
        for _ in range(frame_start):
            world.step()

    # If were saving as a GIF, call get_gif_representation immediately because the process for handling output differs
    if representation == "gif":
        get_gif_representation(world, filepath)
        return

    output = analyzer(world)  # Get image matrix
    img = Image.fromarray(output.astype('uint8'), 'L')  # Save as a grayscale
    if filepath is None:
        # If there is no filepath specified just show it
        img.show()
    else:
        img.save(filepath)


if __name__ == '__main__':
    get_image_map([-0.8979, -0.6356, 1.0000, -0.5669], "gif", filepath="/home/jeremy/Desktop/example.gif")
    print("Done")
