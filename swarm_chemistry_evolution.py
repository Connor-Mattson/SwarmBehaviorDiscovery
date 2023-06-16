"""
Implements the technique shown in Swarm Chemistry:
https://direct.mit.edu/artl/article-abstract/15/1/105/2623/Swarm-Chemistry

Human picks 1-2 out of 8 behaviors shown on a GUI.
Behaviors are subsequently mutated and combined, or just mutated if only one is selected.
In this case, the human acts as the fitness function, searching for novel/interesting behaviors.
For instructions on how to operate this GUI, view `HIL-GUI-manual.pdf`

Author: Jeremy Clark
"""

import math
import os.path
import random
import pandas as pd
import pygame
from collections import namedtuple, deque
from copy import copy, deepcopy
import subprocess
import time
import csv
import gc

from src.ui.button import Button
from data.img_process import get_gif_representation
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
from novel_swarms.sensors.SensorSet import SensorSet
from novel_swarms.world.RectangularWorld import RectangularWorld


class GUIUtils:
    """
    Static utility functions for the GUI
    """
    @staticmethod
    def generate_heterogeneous_world_config(controller):
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
        sensors = SensorSet([BinaryLOSSensor(angle=0)])
        # sensors = SensorSet([BinaryFOVSensor(degrees=True, theta=15)])
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

    @staticmethod
    def generate_world(controller, heterogeneous=False):
        """
        Generates a 500x500 world config using the given controller.

        :param controller: The given controller
        :param heterogeneous: Specifies whether the world is heterogeneous
        :return: A world config
        """
        if heterogeneous:
            return RectangularWorld(GUIUtils.generate_heterogeneous_world_config(controller))
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
        return RectangularWorld(config)

    @staticmethod
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

    @staticmethod
    def _mutate_controller(controller):
        """
        Returns a version of the controller with some random mutations, just like how in nature, offspring have
        genetic mutations. This drives evolution forwards.

        :param controller: The parent controller
        :return: A mutated version of the parent
        """
        mutated_controller = []
        for e in controller:
            new_vel = e + GUIUtils._generate_normal()
            # Make sure the new velocity is between -1 and 1
            while not (-1 < new_vel < 1):
                new_vel = e + GUIUtils._generate_normal()
            mutated_controller.append(new_vel)
        return mutated_controller

    @staticmethod
    def _heterogeneous_crossover(c1, c2):
        """
        Performs crossover on heterogeneous controllers (those with 9 elements).
        Randomly picks a 'target' controller.
        The ratio value (the first value) of the target controller will remain constant.
        Randomly chooses whether to perform crossover on the first or second subspecies of the controller.
        Keeps the other subspecies in the target controller constant.

        The rationale for this approach is that if a heterogeneous genotype has an interesting phenotype,
        the sub-species of that genotype are probably interesting as well, and should be relatively well-preserved.
        :param c1:
        :param c2:
        :return:
        """
        # Pick which controller is the target and which is the supplement
        target_controller = deepcopy(c1) if random.random() > 0.5 else deepcopy(c2)
        supplement_controller = deepcopy(c2) if target_controller == c1 else deepcopy(c1)

        # Mutate both controllers
        target_controller = GUIUtils._mutate_controller(target_controller)
        supplement_controller = GUIUtils._mutate_controller(supplement_controller)

        # Pick subspecies to perform crossover on
        beg_index = 1 if random.random() > 0.5 else 5
        supplement_subspecies = supplement_controller[beg_index:beg_index+4]
        target_subspecies = target_controller[beg_index:beg_index+4]

        # Perform crossover on that subspecies of the target controller, return
        cross_index = math.floor(random.random() * 4)
        supplement_subspecies[cross_index] = target_subspecies[cross_index]
        target_controller[beg_index:beg_index+4] = supplement_subspecies
        return target_controller

    @staticmethod
    def _homogeneous_crossover(c1, c2):
        index = math.floor(random.random() * len(c1))
        rand_val = random.random()
        if rand_val > 0.5:
            offspring = copy(c1)
            offspring[index] = c2[index]
            return offspring
        else:
            offspring = copy(c2)
            offspring[index] = c1[index]
            return offspring

    @staticmethod
    def _crossover(c1, c2):
        if len(c1) == 4:
            return GUIUtils._homogeneous_crossover(c1, c2)
        elif len(c1) == 9:
            return GUIUtils._heterogeneous_crossover(c1, c2)
        else:
            raise ValueError("Controllers must be of length 4 or 9.")

    @staticmethod
    def _breed_controllers(c1, c2):
        """
        Mutates and randomly combines the two given controllers using crossover.
        :return: Offspring of the two controllers
        """
        # Mutate the controllers before combining
        c1 = GUIUtils._mutate_controller(c1)
        c2 = GUIUtils._mutate_controller(c2)
        return GUIUtils._crossover(c1, c2)

    @staticmethod
    def get_new_generation(fit_controllers):
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
            for _ in range(5):
                new_generation.append(GUIUtils._breed_controllers(c1, c2))
            new_generation.append(c1)
            new_generation.append(c2)
            new_generation.append(GUIUtils.generate_random_controllers(1, len(c1))[0])
        # Otherwise, the length of fit_controllers must be 1
        else:
            c = fit_controllers[0]
            for _ in range(6):
                new_generation.append(GUIUtils._mutate_controller(c))
            new_generation.append(c)
            new_generation.append(GUIUtils.generate_random_controllers(1, len(c))[0])
        return new_generation

    @staticmethod
    def truncate_controller(controller):
        """
        Truncate the decimals on the given controller so that it can be displayed conveniently
        """
        return [round(n, 2) for n in controller]

    @staticmethod
    def generate_random_controllers(num=8, length=9, seed=None):
        """
        Generates `num` random controllers of length `length`

        :param num: The number of controllers to generate
        :param length: The length of each controller
        :param seed: Random seed, if any
        :return: A list of randomly generated controllers
        """
        # If we want the function to be deterministic, then seed the random number algorithm
        if seed is not None:
            random.seed(seed)
        controller_list = []
        for _ in range(num):
            controller = [random.uniform(-1, 1) for _ in range(length)]
            magnitudes = [abs(x) for x in controller]
            scaling_factor = 1 / max(magnitudes)
            normalized_controller = [scaling_factor * x for x in controller]
            controller_list.append(normalized_controller)
        return controller_list


class HILGUI:
    """
    Object representing the state of the HIL GUI.
    The GUI consists of `tiles`, which each display a separate simulation.
    For more information about how to operate it, see HIL-GUI-manual.pdf
    """
    def __init__(
            self,
            heterogeneous=False,  # Whether we're representing a heterogeneous swarm
            steps_per_frame=5,  # Timestep interval between screen flips
            time_limit=None,  # The maximum amount of time the user may use.
            click_limit=None,  # The maximum number of steps forward or backward a generation that a user may make.
            save_archived_controllers=False,  # Whether we should save the controllers in the archive as GIFs
            seed=None
    ):
        self.heterogeneous = heterogeneous
        self.steps_per_frame = steps_per_frame
        self.time_limit = time_limit
        self.click_limit = click_limit
        self.save_archived_controllers = save_archived_controllers
        self.seed = seed

        pygame.init()
        self.timesteps = 0  # How many timesteps the current generation has taken
        self.generation = 0  # The current generation of controllers
        self.stats_width = 200  # The width of the stats sidebar (pixels)
        self.width = 2000 + self.stats_width  # The total width of the GUI (pixels)
        self.height = 1000  # The total height of the GUI (pixels)
        # The background screen. Everything else is blitted, directly or indirectly, onto this screen
        self.parent_screen = pygame.display.set_mode((self.width, self.height))
        self.tile_width = 500  # The width of each tile (pixels)
        self.tile_height = 500  # The height of each tile (pixels)
        # The surface that has stats and controls
        self.stats_surface = pygame.Surface((self.stats_width, self.height))
        # The offset of the stats bar relative to the background screen
        stats_offset = (self.width - self.stats_width, 0)
        # Buttons on the stats screen
        self.skip_button = Button("Skip", (60, 25), (5, 150), self.skip, offset=stats_offset)
        self.advance_button = Button("Advance", (60, 25), (5, 180), self.advance, offset=stats_offset)
        self.back_button = Button("Back", (60, 25), (5, 210), self.back, offset=stats_offset)
        self.running = True  # Whether the current generation is running
        self.cycle_output = None  # Collects output at the end of each generation for event handling
        self.tiles = None  # A list of tile objects corresponding to this generation
        # A stack where elements are lists of controllers. Top = current generation, bottom = first generation
        self.controller_history = deque()
        self.fit_controllers = []  # Controllers selected for evolution
        self.start_time = time.time()  # The time when the GUI was opened (seconds)
        self.number_of_clicks = 0  # How many times the user has gone forward or back
        self.archived_controllers = []  # Controllers to be saved
        pygame.font.init()
        font = pygame.font.Font(None, 25)
        self.white_color = (255, 255, 255)
        self.generation_text = font.render(f"Current generation: {self.generation}", True, self.white_color)
        self.click_text = font.render(f"Number of clicks: {self.number_of_clicks}", True, self.white_color)
        del font
        pygame.font.quit()

        self.absolute_positions_by_index = [
            (0, 0),
            (500, 0),
            (1000, 0),
            (1500, 0),
            (0, 500),
            (500, 500),
            (1000, 500),
            (1500, 500)
        ]

    def get_tiles_from_controllers(self, controller_list):
        """
        Utility function for displaying tiles on the GUI.
        :param controller_list: A list of controllers.
        :return: The tiles corresponding to the controllers of `controller_list`
        """
        # Indices of list correspond to the tile index, which increases from top to bottom and left to right
        # Used for putting the tiles in the right place on the GUI
        tiles = [
            GUITile(controller_list[i], i, self.absolute_positions_by_index[i], self)
            for i in range(8)
        ]
        return tiles

    def print_archived_controllers(self):
        """
        Prints the list of controllers that have been marked for archive during this generation.
        Does NOT print the entire list of controllers that have ever been marked for archive.
        :return: A list of truncated controllers
        """
        for tile in self.tiles:
            if tile.archived:
                truncated_controller = GUIUtils.truncate_controller(tile.controller)
                print(truncated_controller)

    def skip(self):
        """
        Runs when the Skip button is pressed.
        """
        self.running = False
        self.cycle_output = "Skip"

    def back(self):
        """
        Runs when the Back button is pressed.
        """
        # We cannot go back any further if the generation <= 0
        if self.generation > 0:
            self.running = False
            self.cycle_output = "Back"

    def advance(self):
        """
        Runs when the Advance button is pressed.
        """
        fit_controllers = self.fit_controllers
        # If the number of controllers selected for evolution is not 1 or 2, don't do anything
        if len(fit_controllers) in (1, 2):
            self.running = False
            self.cycle_output = fit_controllers

    def get_time_string(self):
        """
        :return: The time since the GUI was opened is H:M:S format
        """
        current_time = time.time()
        seconds = current_time - self.start_time
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        # TODO: Add placeholder zeroes
        return f"{int(hours)}:{int(minutes)}:{int(seconds)}"

    def draw_stats_surface(self):
        """
        Helper function for drawing the main GUI. Draws the sidebar.
        """
        pygame.font.init()
        font = pygame.font.Font(None, 25)
        timestep_text = font.render(f"Timesteps: {self.timesteps}", True, self.white_color)
        time_text = font.render(self.get_time_string(), True, self.white_color)
        del font
        pygame.font.quit()
        self.stats_surface.fill((0, 0, 0))
        text_x, text_y = 0, 20
        self.stats_surface.blit(self.generation_text, (text_x, text_y))
        text_y += 30
        self.stats_surface.blit(self.click_text, (text_x, text_y))
        text_y += 30
        self.stats_surface.blit(timestep_text, (text_x, text_y))
        text_y += 30
        self.stats_surface.blit(time_text, (text_x, text_y))

        del timestep_text
        del time_text

        self.skip_button.draw(self.stats_surface)
        self.advance_button.draw(self.stats_surface)
        self.back_button.draw(self.stats_surface)
        self.parent_screen.blit(self.stats_surface, (self.width - self.stats_width, 0))

    def draw(self):
        """
        Draws the current frame of the GUI.
        """
        self.parent_screen.fill((0, 0, 0))
        # Draw every tile
        for tile in self.tiles:
            tile.draw()
        self.draw_stats_surface()
        self.stats_surface.blit(self.parent_screen, (self.width-self.stats_width, 0))
        pygame.display.flip()

    def _step_all_worlds(self):
        """
        Advances all the current worlds in the generation by one step.
        """
        for tile in self.tiles:
            tile.world.step()

    def _one_cycle(self):
        """
        Runs one generation of simulation.
        :return: The termination flag associated with this generation. Tells driver function how to handle it.
        """
        # Get the current generation of controllers off of the history stack.
        current_controllers = self.controller_history[-1]
        self.tiles = self.get_tiles_from_controllers(current_controllers)

        while self.running:
            # Event handling
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    self.cycle_output = "Quit"
                    return
            self.back_button.listen(events)
            self.skip_button.listen(events)
            self.advance_button.listen(events)
            for tile in self.tiles:
                tile.evolve_button.listen(events)
                tile.save_button.listen(events)

            self._step_all_worlds()  # Advance every world by one timestep
            self.timesteps += 1
            pygame.event.pump()  # Prevent force quit window from appearing

            # If it has been `steps_per_frame` timesteps since we last drew, update the screen
            if self.timesteps % self.steps_per_frame == 0:
                self.draw()

        # If this block was reached, it means that the user did not try to quit the GUI
        self.number_of_clicks += 1
        self.timesteps = 0
        self.running = True  # Set this back to True for the next generation

    def run(self):
        """
        Driver function for program execution.
        """
        # Ensure that the first set of choices is always the same for every user
        random_controllers = GUIUtils.generate_random_controllers(seed=self.seed)
        self.controller_history.append(random_controllers)

        while True:
            # Handle limit breaching
            if self.click_limit is not None and self.number_of_clicks >= self.click_limit:
                print(f"Click limit of {self.click_limit} clicks reached.")
                return
            elapsed_time = self.get_time_string()
            if self.time_limit is not None and elapsed_time >= self.time_limit:
                print(f"Time limit of {elapsed_time} reached.")

            # If there is nothing in the controller history, add a list of randomly-generated controllers
            if len(self.controller_history) == 0:
                random_controllers = GUIUtils.generate_random_controllers()
                self.controller_history.append(random_controllers)
            self._one_cycle()

            # If the user pressed "Advance", cycle_output is set to the list of controllers to advance with
            if type(self.cycle_output) == list:
                controller_list = GUIUtils.get_new_generation(self.cycle_output)
                self.controller_history.append(controller_list)
                self.generation += 1
            # If the user quits, print the controllers marked for archive this generation and break.
            elif self.cycle_output == "Quit":
                self.print_archived_controllers()
                break
            # If the user skips, randomly generate a new controller list for the next generation
            elif self.cycle_output == "Skip":
                self.print_archived_controllers()
                controller_list = GUIUtils.generate_random_controllers()
                self.controller_history.append(controller_list)
                self.generation += 1
            # If the user wants to go back, pop a controller list off the controller history stack
            elif self.cycle_output == "Back" and self.generation > 0:
                self.print_archived_controllers()
                self.controller_history.pop()
                self.generation -= 1
            # Handle wrong flag type
            else:
                raise ValueError(f"Received an improper flag from HILGUI.cycle_output. Got {self.cycle_output}.")

            pygame.font.init()
            font = pygame.font.Font(None, 25)
            self.generation_text = font.render(f"Current generation: {self.generation}", True, self.white_color)
            self.click_text = font.render(f"Number of clicks: {self.number_of_clicks}", True, self.white_color)
            pygame.font.quit()

            for tile in self.tiles:
                tile.close()
                del tile
            self.tiles = None
            gc.collect()

        # Print stats
        print("Simulation terminated. We hope to see you again!")
        print(f"Number of clicks: {self.number_of_clicks}")
        print(f"Time elapsed: {self.get_time_string()}")


class GUITile:
    def __init__(self, controller, index, abs_pos, parent_gui):
        self.controller = controller
        self.index = index
        self.abs_pos = abs_pos
        self.gui = parent_gui

        self.world = GUIUtils.generate_world(controller, heterogeneous=parent_gui.heterogeneous)
        self.tile_surface = pygame.Surface((self.gui.tile_width, self.gui.tile_height))
        save_button_coords = (self.gui.tile_width - 70, self.gui.tile_height - 25)
        evolve_button_coords = (self.gui.tile_width - 145, self.gui.tile_height - 25)
        self.evolve_button = Button(
            "Evolve",
            (70, 25),
            evolve_button_coords,
            self.evolve_button_pressed,
            self.get_tile_offset()
        )
        self.save_button = Button(
            "Save",
            (70, 25),
            save_button_coords,
            self.save_button_pressed,
            self.get_tile_offset()
        )
        self.highlighted = False
        self.archived = False
        pygame.font.init()
        font = pygame.font.Font(None, 18)
        truncated_controller = GUIUtils.truncate_controller(self.controller)
        self.text = font.render(f"Params: {truncated_controller}", True, (0, 0, 0))
        pygame.font.quit()

    def get_tile_offset(self):
        absolute_positions_by_index = [
            (0, 0),
            (500, 0),
            (1000, 0),
            (1500, 0),
            (0, 500),
            (500, 500),
            (1000, 500),
            (1500, 500)
        ]
        return absolute_positions_by_index[self.index]

    def save_button_pressed(self):
        if self.archived:
            self.archived = False
            self.save_button.bg_color = (150, 150, 150)
        else:
            self.archived = True
            self.save_button.bg_color = (0, 150, 0)
            self.gui.archived_controllers.append(deepcopy(self.controller))
            filepath = "data/interesting-controllers.csv"
            if self.gui.save_archived_controllers:
                with open(filepath, 'a', newline='') as file:
                    writer = csv.writer(file)
                    # Write the new row to the CSV file
                    writer.writerow(self.controller)

    def evolve_button_pressed(self):
        if self.highlighted:
            self.highlighted = False
            self.gui.fit_controllers.remove(self.controller)
            self.evolve_button.bg_color = (150, 150, 150)
        else:
            self.highlighted = True
            self.gui.fit_controllers.append(deepcopy(self.controller))
            self.evolve_button.bg_color = (0, 150, 0)
            if len(self.gui.fit_controllers) > 2:
                for tile in self.gui.tiles:
                    tile.highlighted = False
                    tile.evolve_button.bg_color = (150, 150, 150)
                self.gui.fit_controllers = []

    def draw(self):
        self.tile_surface.fill((0, 0, 0))
        self.world.draw(self.tile_surface)
        pygame.draw.rect(self.tile_surface, (255, 255, 255), (0, 0, self.gui.tile_width-50, 15))
        self.tile_surface.blit(self.text, (0, 0))
        self.save_button.draw(self.tile_surface)
        self.evolve_button.draw(self.tile_surface)

        parent_screen = self.gui.parent_screen
        parent_screen.blit(self.tile_surface, self.abs_pos)

    def close(self):
        self.gui = None
        del self


def configuration_screen():
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


def update_manual_pdf():
    md_file_path = "HIL-GUI-manual.md"
    pdf_file_path = "HIL-GUI-manual.pdf"
    try:
        subprocess.run(['pandoc', md_file_path, '-o', pdf_file_path])
    except OSError:
        print(f"{md_file_path} does not exist.")


def render_interesting_worlds():
    interesting_behaviors_dirpath = "data/interesting-behaviors/"
    world_cache_filepath = "data/interesting-controllers.csv"
    df = pd.read_csv(world_cache_filepath, header=None)
    for i, row in df.iterrows():
        controller = list(row)
        print(f"Number {i}. Controller = {controller}")
        gif_filename = f"{controller}.gif"
        gif_filepath = os.path.join(interesting_behaviors_dirpath, gif_filename)
        world = RectangularWorld(GUIUtils.generate_heterogeneous_world_config(controller))
        for _ in range(2300):
            world.step()
        get_gif_representation(world, gif_filepath, steps=1000)


if __name__ == '__main__':
    # gui = HILGUI(heterogeneous=True, save_archived_controllers=True, seed=2)
    # gui.run()
    # REMEMBER TO CLEAR
    render_interesting_worlds()
