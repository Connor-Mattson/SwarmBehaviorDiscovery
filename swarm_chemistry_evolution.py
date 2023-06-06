"""
Implements the technique shown in Swarm Chemistry:
https://direct.mit.edu/artl/article-abstract/15/1/105/2623/Swarm-Chemistry

Human picks 1-2 out of 6 behaviors shown on a GUI.
Behaviors are subsequently mutated and combined, or just mutated if only one is selected.
In this case, the human acts as the fitness function, searching for novel/interesting behaviors.

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
import psutil
import csv

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

    @staticmethod
    def _breed_controllers(c1, c2):
        """
        Mutates and randomly combines the two given controllers using crossover.
        :return: Offspring of the two controllers
        """
        # Mutate the controllers before combining
        c1 = GUIUtils._mutate_controller(c1)
        c2 = GUIUtils._mutate_controller(c2)
        crossover_index = math.floor(random.random() * len(c1))
        return GUIUtils._crossover(crossover_index, c1, c2, random.random())

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
        Truncate the decimals on the given controller so that it can be displayed
        """
        return [round(n, 2) for n in controller]

    @staticmethod
    def generate_random_controllers(num=8, length=9):
        """
        Generates `num` random controllers of length `length`

        :param num: The number of controllers to generate
        :param length: The length of each controller
        :return: A list of randomly generated controllers
        """
        controller_list = []
        for _ in range(num):
            controller = [random.uniform(-1, 1) for _ in range(length)]
            magnitudes = [abs(x) for x in controller]
            scaling_factor = 1 / max(magnitudes)
            normalized_controller = [scaling_factor * x for x in controller]
            controller_list.append(normalized_controller)
        return controller_list


class HILGUI:
    def __init__(
            self,
            heterogeneous=False,
            steps_per_frame=5,
            time_limit=None,
            click_limit=None,
            save_archived_controllers=False,
    ):
        pygame.init()
        self.steps_per_frame = steps_per_frame
        self.timesteps = 0
        self.generation = 0
        self.stats_width = 200
        self.width = 2000 + self.stats_width
        self.height = 1000
        self.parent_screen = pygame.display.set_mode((self.width, self.height))
        self.tile_width = 500
        self.tile_height = 500
        self.stats_surface = pygame.Surface((self.stats_width, self.height))
        offset = (self.width - self.stats_width, 0)
        self.skip_button = Button("Skip", (60, 25), (5, 150), self.skip, offset=offset)
        self.advance_button = Button("Advance", (60, 25), (5, 180), self.advance, offset=offset)
        self.back_button = Button("Back", (60, 25), (5, 210), self.back, offset=offset)
        self.running = True
        self.cycle_output = None
        self.tiles = None
        self.controller_history = deque()
        self.fit_controllers = []
        self.heterogeneous = heterogeneous
        self.start_time = None
        self.time_limit = time_limit
        self.click_limit = None
        self.number_of_clicks = 0
        self.save_archived_controllers = save_archived_controllers
        self.click_limit = click_limit
        self.cpu_threshold = 50
        self.archived_controllers = []

    def get_tiles_from_controllers(self, controller_list):
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
        tiles = [
            GUITile(controller_list[i], i, absolute_positions_by_index[i], self)
            for i in range(8)
        ]
        return tiles

    def print_archived_controllers(self):
        for tile in self.tiles:
            if tile.archived:
                truncated_controller = GUIUtils.truncate_controller(tile.controller)
                print(truncated_controller)

    def skip(self):
        self.running = False
        self.cycle_output = "Skip"

    def back(self):
        if self.generation > 0:
            self.running = False
            self.cycle_output = "Back"

    def advance(self):
        fit_controllers = self.fit_controllers
        if len(fit_controllers) in (1, 2):
            self.running = False
            self.cycle_output = fit_controllers

    def get_time_string(self):
        current_time = time.time()
        seconds = current_time - self.start_time
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours)}:{int(minutes)}:{int(seconds)}"

    def draw(self):
        self.parent_screen.fill((0, 0, 0))
        for tile in self.tiles:
            tile.draw()
        self.stats_surface.fill((0, 0, 0))
        text_x, text_y = 0, 20
        pygame.font.init()
        font = pygame.font.Font(None, 25)
        white_color = (255, 255, 255)
        text = font.render(f"Current generation: {self.generation}", True, white_color)
        self.stats_surface.blit(text, (text_x, text_y))
        text_y += 30
        text = font.render(f"Timesteps: {self.timesteps}", True, white_color)
        self.stats_surface.blit(text, (text_x, text_y))
        text_y += 30
        text = font.render(self.get_time_string(), True, white_color)
        self.stats_surface.blit(text, (text_x, text_y))
        text_y += 30
        text = font.render(f"Number of clicks: {self.number_of_clicks}", True, white_color)
        self.stats_surface.blit(text, (text_x, text_y))
        pygame.font.quit()

        self.skip_button.draw(self.stats_surface)
        self.advance_button.draw(self.stats_surface)
        self.back_button.draw(self.stats_surface)
        self.parent_screen.blit(self.stats_surface, (self.width - self.stats_width, 0))
        pygame.display.flip()

    def step_all_worlds(self):
        for tile in self.tiles:
            tile.world.step()

    def one_cycle(self):
        current_controllers = self.controller_history[-1]
        self.tiles = self.get_tiles_from_controllers(current_controllers)
        psutil.cpu_percent(interval=None, percpu=True)
        while self.running:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    self.cycle_output = "Quit"
                    return
            # Listen for button clicks
            self.back_button.listen(events)
            self.skip_button.listen(events)
            self.advance_button.listen(events)
            for tile in self.tiles:
                tile.evolve_button.listen(events)
                tile.save_button.listen(events)

            self.step_all_worlds()
            self.timesteps += 1
            pygame.event.pump()  # Prevent force quit window from appearing

            # If it has been `steps_per_frame` timesteps since we last drew, update the screen
            if self.timesteps % self.steps_per_frame == 0:
                self.draw()

        self.number_of_clicks += 1
        self.timesteps = 0
        self.running = True

    def run(self):
        self.start_time = time.time()
        while True:
            if self.click_limit is not None and self.number_of_clicks >= self.click_limit:
                print(f"Click limit of {self.click_limit} clicks reached.")
                return
            elapsed_time = self.get_time_string()
            if self.time_limit is not None and elapsed_time >= self.time_limit:
                print(f"Time limit of {elapsed_time} reached.")
            if len(self.controller_history) == 0:
                random_controllers = GUIUtils.generate_random_controllers()
                self.controller_history.append(random_controllers)
            self.cycle_output = "_"
            self.one_cycle()
            if type(self.cycle_output) == list:
                controller_list = GUIUtils.get_new_generation(self.cycle_output)
                self.controller_history.append(controller_list)
                self.generation += 1
            elif self.cycle_output == "Quit":
                self.print_archived_controllers()
                break
            elif self.cycle_output == "Skip":
                self.print_archived_controllers()
                controller_list = GUIUtils.generate_random_controllers()
                self.controller_history.append(controller_list)
                self.generation += 1
            elif self.cycle_output == "Back" and self.generation > 0:
                self.print_archived_controllers()
                self.controller_history.pop()
                self.generation -= 1
            else:
                raise ValueError(f"Received an improper flag from HILGUI.cycle_output. Got {self.cycle_output}.")

        print("Simulation terminated.")
        print(f"Number of clicks: {self.number_of_clicks}")
        print(f"Time elapsed: {self.get_time_string()}")


class GUITile:

    def __init__(self, controller, index, abs_pos, parent_gui):
        self.gui = parent_gui
        self.controller = controller
        self.index = index
        self.abs_pos = abs_pos
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
            self.gui.archived_controllers.append(self.controller)
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
            self.gui.fit_controllers.append(self.controller)
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
        pygame.font.init()
        font = pygame.font.Font(None, 18)
        truncated_controller = GUIUtils.truncate_controller(self.controller)
        text = font.render(f"Params: {truncated_controller}", True, (0, 0, 0))
        pygame.font.quit()
        self.tile_surface.blit(text, (0, 0))
        self.save_button.draw(self.tile_surface)
        self.evolve_button.draw(self.tile_surface)

        parent_screen = self.gui.parent_screen
        parent_screen.blit(self.tile_surface, self.abs_pos)


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
        for _ in range(1200):
            world.step()
        get_gif_representation(world, gif_filepath, steps=1000)


if __name__ == '__main__':
    # gui = HILGUI(heterogeneous=True, save_archived_controllers=True)
    # gui.run()
    render_interesting_worlds()
