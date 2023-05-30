"""
Implements the technique shown in Swarm Chemistry:
https://direct.mit.edu/artl/article-abstract/15/1/105/2623/Swarm-Chemistry

Human picks 1-2 out of 6 behaviors shown on a GUI.
Behaviors are subsequently mutated and combined, or just mutated if only one is selected.
In this case, the human acts as the fitness function, searching for novel/interesting behaviors.

Author: Jeremy Clark
"""
import math
import random
import pygame
from collections import namedtuple, deque
from copy import copy
import pygame_gui

from novel_swarms.config.AgentConfig import DiffDriveAgentConfig
from novel_swarms.config.AgentConfig import DroneAgentConfig
from novel_swarms.config.AgentConfig import UnicycleAgentConfig
from novel_swarms.config.AgentConfig import LevyAgentConfig
from novel_swarms.config.AgentConfig import MazeAgentConfig
from novel_swarms.config.AgentConfig import StaticAgentConfig
from novel_swarms.config.HeterogenSwarmConfig import HeterogeneousSwarmConfig
from novel_swarms.config.WorldConfig import RectangularWorldConfig
from novel_swarms.sensors.BinaryFOVSensor import BinaryFOVSensor
from novel_swarms.sensors.BinaryLOSSensor import BinaryLOSSensor
from novel_swarms.sensors.GenomeDependentSensor import GenomeFOVSensor
from novel_swarms.sensors.SensorSet import SensorSet
from novel_swarms.world.RectangularWorld import RectangularWorld


def _generate_heterogeneous_world_config(controller):
    """
    Helper function for _generate_world_config

    :param controller: The given 9-element long controller
    :return: A world config for heterogeneous controllers
    """
    pop = 30  # Population of agents
    rat = abs(controller[0])  # Ratio of agent 1 to agent 2
    # Calculate populations of the two groups
    p1 = math.floor(rat * pop)
    p2 = pop - p1
    # Controllers of the two groups are embedded in the parent controller
    c1, c2 = controller[1:5], controller[5:9]
    # sensors = SensorSet([BinaryLOSSensor(angle=0)])
    sensors = SensorSet([BinaryFOVSensor(degrees=True, theta=15)])
    a1_config = DiffDriveAgentConfig(controller=c1, sensors=sensors, seed=None, body_color=(200, 0, 0))
    a2_config = DiffDriveAgentConfig(controller=c2, sensors=sensors, seed=None, body_color=(0, 200, 0))
    agent_config = HeterogeneousSwarmConfig()
    agent_config.add_sub_populuation(a1_config, p1)
    agent_config.add_sub_populuation(a2_config, p2)
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


def _generate_world_config(controller, heterogeneous=False):
    """
    Generates a 500x500 world config using the given controller.

    :param controller: The given controller
    :param heterogeneous: Specifies whether the world is hetergeneous
    :return: A world config
    """
    if heterogeneous:
        return _generate_heterogeneous_world_config(controller)
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


def _generate_random_controllers(num=6, length=4):
    """
    Generates `num` random controllers of length `length`

    :param num: The number of controllers to generate
    :param length: The length of each controller
    :return: A list of randomly generated controllers
    """
    controller_list = []
    for _ in range(num):
        controller = [random.uniform(-1, 1) for _ in range(length)]
        controller_list.append(controller)
    return controller_list


def _calculate_tile_clicked(mouse_pos):
    """
    Given the position of a mouse click, returns the 'index'  of the tile clicked.
    Upper right -> upper-left -> lower right -> lower left
    Upper right index = 0
    Lower left index = 5
    There are 6 tiles total.

    :param mouse_pos: The position of the mouse on the screen at the time of the event
    :return: The index of the tile that was clicked
    """
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
    # If the user didn't click on one of the tiles
    if tile_pos not in pos_dict.keys():
        return False
    return pos_dict[tile_pos]


def _detect_double_click(current_time, last_click_time):
    """
    Given the current time and the time of the previous click, return whether a double click
    event has occurred.
    :param current_time: The current time in milliseconds
    :param last_click_time: The time of the last click in milliseconds
    :return: Whether a double click event has occurred
    """
    if last_click_time is None:
        return False
    return current_time - last_click_time < 500


def _truncate_controller(controller):
    """
    Truncate the decimals on the given controller so that it can be displayed
    """
    return [round(n, 2) for n in controller]


def _step_tile_world(w):
    """
    Helper function for stepping worlds simultaneously.
    """
    w.step()
    return w


def HIL_evolution_GUI(controllers, heterogeneous=False, generation=0):
    """
    Given a list of controllers, displays the corresponding behaviors on a layout of tiles.
    The user may select one or two tiles, which are then returned as the "fittest" of the group.
    The user may also press 's' while hovering their mouse over a tile to print its controller to the console
    if they find the behavior shown in that tile to be particularly interesting.

    :param controllers: The list of controllers that are simulated in the tiles
    :param heterogeneous: Whether the simulations are heterogeneous
    :param generation: How many rounds of evolution the user has gone through
    :return: The "fittest" controllers
    """
    timesteps = 0  # The current number of steps taken
    # The worlds corresponding to the controllers
    worlds = [RectangularWorld(_generate_world_config(e, heterogeneous=heterogeneous)) for e in controllers]

    pygame.init()

    steps_per_frame = 5  # How many steps the worlds should take per frame
    sidebar_width = 200
    gui_width = 1500 + sidebar_width  # 200 extra pixels in width to show stats
    gui_height = 1000
    tile_height = 500
    tile_width = 500
    parent_screen = pygame.display.set_mode((gui_width, gui_height))
    pygame.display.set_caption("Swarm Chemistry HIL-assisted Evolution")

    # The positions of the top left corners of the tiles
    tile_positions = [
        (0, 0),       # 0
        (500, 0),     # 1
        (1000, 0),    # 2
        (0, 500),     # 3
        (500, 500),   # 4
        (1000, 500)   # 5
    ]

    manager = pygame_gui.UIManager((gui_width, gui_height))
    skip_button = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(gui_width - sidebar_width + 5, 100, 60, 25),
        text="Skip",
        manager=manager,
    )
    advance_button = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(gui_width - sidebar_width + 5, 130, 85, 25),
        text="Advance",
        manager=manager
    )
    back_button = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(gui_width - sidebar_width + 5, 160, 60, 25),
        text="Back",
        manager=manager
    )

    # The Pygame surfaces corresponding to each tile
    tiles = []
    tile_images = []
    for i in range(6):
        tile = pygame.Surface((tile_width, tile_height))
        tile_image = pygame_gui.elements.UIImage(
            pygame.Rect(*tile_positions[i], tile_width, tile_height),
            tile,
            manager
        )
        tile_images.append(tile_image)
        tiles.append(tile)

    highlighted_indices = set()  # The indices corresponding to the tiles that have been clicked
    archived_controllers = list()  # The indices corresponding to the tiles that have been archived

    clock = pygame.time.Clock()
    last_click_time = None
    running = True
    while running:
        time_delta = clock.tick(60) / 1000.0
        current_time = pygame.time.get_ticks()

        # Handle if user tries to quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                for c in archived_controllers:
                    print(_truncate_controller(controllers[c]))
                return "stop"
            elif event.type == pygame.MOUSEBUTTONUP and event.button == pygame.BUTTON_LEFT:
                if _detect_double_click(current_time, last_click_time):
                    for j, tile_image in enumerate(tile_images):
                        if tile_image.rect.collidepoint(event.pos):
                            archived_controllers.append(j)
                else:
                    for j, tile_image in enumerate(tile_images):
                        if tile_image.rect.collidepoint(event.pos):
                            if j in highlighted_indices:
                                highlighted_indices.remove(j)
                            else:
                                highlighted_indices.add(j)
                    if len(highlighted_indices) > 2:
                        highlighted_indices = set()
                last_click_time = current_time
            # If the user clicks on a button
            elif event.type == pygame.USEREVENT:
                if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                    # If the user doesn't find anything interesting, skip this round
                    if event.ui_element == skip_button:
                        for c in archived_controllers:
                            print(_truncate_controller(controllers[c]))
                        return "skip"
                    elif event.ui_element == advance_button:
                        # If the user has selected one or two tiles, return the fit controllers
                        if len(highlighted_indices) in (1, 2):
                            fit_controllers = [controllers[j] for j in highlighted_indices]
                            pygame.quit()
                            for c in archived_controllers:
                                print(_truncate_controller(controllers[c]))
                            return fit_controllers
                    elif event.ui_element == back_button and generation != 1:
                        for c in archived_controllers:
                            print(_truncate_controller(controllers[c]))
                        return "back"
            manager.process_events(event)

        # Sidebar displays stats about the simulation
        parent_screen.fill((0, 0, 0))

        stats_surface = pygame.Surface((200, 1000))
        stats_surface.fill((0, 0, 0))
        text_x, text_y = 0, 20
        font = pygame.font.Font(None, 25)
        text = font.render(f"Current generation: {generation}", True, (255, 255, 255))
        stats_surface.blit(text, (text_x, text_y))
        text_y += 30
        text = font.render(f"Timesteps: {timesteps}", True, (255, 255, 255))
        stats_surface.blit(text, (text_x, text_y))
        parent_screen.blit(stats_surface, (1500, 0))

        timesteps += 1  # Increment the timer

        # For each index corresponding to a tile, step and draw
        for i in range(6):
            current_tile = tiles[i]
            tile_world = worlds[i]
            tile_world.step()
            if timesteps % steps_per_frame == 0:
                current_tile.fill((0, 0, 0))
                tile_world.draw(current_tile)
                pygame.draw.rect(current_tile, (255, 255, 255), (0, 0, 375, 15))
                font = pygame.font.Font(None, 18)
                truncated_controller = _truncate_controller(controllers[i])
                text = font.render(f"Params: {truncated_controller}", True, (0, 0, 0))
                current_tile.blit(text, (0, 0))
                if i in archived_controllers:
                    text = font.render("Save", True, (0, 0, 0))
                    pygame.draw.rect(current_tile, (0, 150, 0), (tile_width - 35, 0, 35, 20))
                    current_tile.blit(text, (tile_width - 30, 4))
                if i in highlighted_indices:
                    text = font.render("Evolve", True, (0, 0, 0))
                    pygame.draw.rect(current_tile, (173, 216, 230), (tile_width - 70, 0, 35, 20))
                    current_tile.blit(text, (tile_width - 70, 4))
                tile_images[i].set_image(current_tile)

        manager.update(time_delta)
        manager.draw_ui(parent_screen)

        # Flip the screen if we haven't done so in a while
        if timesteps % steps_per_frame == 0:
            pygame.display.flip()

    pygame.quit()


def _generate_normal(std_dev=0.2):
    """
    Generates a random number on a distribution curve centered at 0 and with a specified standard deviation.

    :param std_dev: The standard deviation
    :return: Normalized random number
    """
    u1 = random.random()
    u2 = random.random()
    # Box-Muller transformation
    z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    x = std_dev * z
    return x


def _mutate_controller(controller):
    """
    Returns a version of the controller with some random mutations, just like how in nature, offspring have
    genetic mutations. This drives evolution forwards.

    :param controller: The parent controller
    :return: A mutated version of the parent
    """
    mutated_controller = []
    for e in controller:
        new_vel = e + _generate_normal()
        # Make sure the new velocity is between -1 and 1
        while not (-1 < new_vel < 1):
            new_vel = e + _generate_normal()
        mutated_controller.append(new_vel)
    return mutated_controller


def _crossover(index, c1, c2, rand_val):
    """
    Performs crossover on the given two controllers at the given index.
    If rand_val > 0.5, then c1 is returned with a swap at `index`.
    Otherwise, c2 is returned with a swap at `index`.
    Does not mutate any of the parameters.
    :return: A "crossed-over" version of one of the controllers
    """
    if rand_val > 0.5:
        offspring = copy(c1)
        offspring[index] = c2[index]
        return offspring
    else:
        offspring = copy(c2)
        offspring[index] = c1[index]
        return offspring


def _breed_controllers(c1, c2):
    """
    Mutates and randomly combines the two given controllers using crossover.
    :return: Offspring of the two controllers
    """
    # Mutate the controllers before combining
    c1 = _mutate_controller(c1)
    c2 = _mutate_controller(c2)
    crossover_index = math.floor(random.random() * len(c1))
    return _crossover(crossover_index, c1, c2, random.random())


def _get_new_generation(fit_controllers):
    """
    Given a list of fit controllers, generates a new generation of controllers that's the result
    of mutating and/or combining the controllers.
    `fit_controllers` can be either 1 or 2 controllers long.
    If it's 1 controller long, randomly mutate it 5 times

    :param fit_controllers: The given controllers to make a new generation out of
    :return: A list of controllers representing the next generation in the evolution cycle
    """
    new_generation = []
    # If the user has selected two controllers
    if len(fit_controllers) == 2:
        c1, c2 = fit_controllers
        new_generation.append(c1)
        new_generation.append(c2)
        for _ in range(4):
            new_generation.append(_breed_controllers(c1, c2))
    # Otherwise, the length of fit_controllers must be 1
    else:
        c = fit_controllers[0]
        new_generation.append(c)
        for _ in range(5):
            new_generation.append(_mutate_controller(c))
    return new_generation


def _selection_screen():
    """
    A screen that displays before the user does HIL-assisted evolution.
    It allows them to change the settings, such as the capability model used, the type of sensors, etc.

    :return: The selected settings
    """
    pygame.init()
    screen = pygame.display.set_mode((500, 500))

    selection_line = namedtuple('selection_line', "prompt choices")
    model_list = [
        DiffDriveAgentConfig,
        DroneAgentConfig,
        LevyAgentConfig,
        MazeAgentConfig,
        StaticAgentConfig,
        UnicycleAgentConfig
    ]
    sensor_list = [BinaryLOSSensor, BinaryFOVSensor]
    lines = [
        selection_line("Capability Model", model_list),
        selection_line("Sensor", sensor_list)
             ]
    indices = [0, 0]
    row, col = 0, 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    col += 1
                    indices[row] += 1
                elif event.key == pygame.K_LEFT:
                    col -= 1
                    indices[row] -= 1
                elif event.key == pygame.K_RETURN:
                    row += 1
                    col = 0
        if row >= len(lines):
            running = False
        screen.fill((0, 0, 0))
        font = pygame.font.Font(None, 20)
        text = font.render("Use arrows to toggle options, ENTER to select", True, (200, 200, 200))
        screen.blit(text, (0, 0))
        font = pygame.font.Font(None, 30)
        for i in range(len(lines)):
            line = lines[i]
            line_color = (200, 200, 200)
            if i < row:
                line_color = (0, 150, 0)
            current_choice = line.choices[indices[i] % len(line.choices)].__qualname__
            text = font.render(f"{line.prompt} = {current_choice}", True, line_color)
            x, y = 5, (i + 1) * 25
            screen.blit(text, (x, y))
        pygame.display.flip()

    settings = []
    for i, line in enumerate(lines):
        settings.append(lines[i].choices[indices[i] % len(lines[i].choices)])
    return settings


def HIL_evolution():
    """
    Driver function for Human in the Loop evolution.
    """
    controllers = _generate_random_controllers(length=9)
    history_stack = deque()
    history_stack.append(controllers)

    generation = 0
    while True:
        cycle_output = HIL_evolution_GUI(controllers, generation=generation, heterogeneous=False)
        if cycle_output == "skip":
            controllers = _generate_random_controllers(length=9)
            generation += 1
            history_stack.append(controllers)
        elif cycle_output == "back":
            controllers = history_stack.pop()
            generation -= 1
        elif cycle_output == "stop":
            return
        # The user pressed the advance button
        else:
            fit_controllers = cycle_output
            controllers = _get_new_generation(fit_controllers)
            history_stack.append(controllers)
            generation += 1


if __name__ == '__main__':
    HIL_evolution()
