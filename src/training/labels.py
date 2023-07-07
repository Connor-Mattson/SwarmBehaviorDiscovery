import os
import numpy as np

class LabelSet:
    def __init__(self, file_path):
        """
        Initialize an instance of LabelSet
        :param file_path: the path to a file containing labeled values
        """
        self.labels = []
        self.class_types = set()
        with open(file_path, "r") as f:
            lines = f.readlines()
            self.labels = [-1 for _ in range(len(lines))]
            for line in lines:
                triplet = self.csv_line_to_vec(line)
                self.labels[int(triplet[0])] = int(triplet[1])
                self.class_types.add(int(triplet[1]))
        self.color_map = self.get_color_mapping()

    def __getitem__(self, key):
        return self.labels[key]

    def get_labels(self):
        return self.labels

    def get_colors(self):
        return [self.color_map[i] for i in self.labels]

    def set_color(self, index, color):
        self.color_map[index] = color

    def get_color_mapping(self, as_keys=False):
        if as_keys:
            color_pallete = list(self._kelly_colors().keys())
        else:
            color_pallete = self.normalized_colors()
        return {i: color_pallete[i % len(color_pallete)] for i in range(len(self.class_types))}

    def normalized_colors(self):
        color_pallete = list(self._kelly_colors().values())
        for i in range(len(color_pallete)):
            color_pallete[i] = (color_pallete[i][0] / 255, color_pallete[i][1] / 255, color_pallete[i][2] / 255)
        return color_pallete

    def _kelly_colors(self):
        return dict(vivid_yellow=(255, 179, 0),
                    strong_purple=(128, 62, 117),
                    vivid_orange=(255, 104, 0),
                    very_light_blue=(166, 189, 215),
                    vivid_red=(193, 0, 32),
                    grayish_yellow=(206, 162, 98),
                    medium_gray=(129, 112, 102),

                    # these aren't good for people with defective color vision:
                    vivid_green=(0, 125, 52),
                    strong_purplish_pink=(246, 118, 142),
                    strong_blue=(0, 83, 138),
                    strong_yellowish_pink=(255, 122, 92),
                    strong_violet=(83, 55, 122),
                    vivid_orange_yellow=(255, 142, 0),
                    strong_purplish_red=(179, 40, 81),
                    vivid_greenish_yellow=(244, 200, 0),
                    strong_reddish_brown=(127, 24, 13),
                    vivid_yellowish_green=(147, 170, 0),
                    deep_yellowish_brown=(89, 51, 21),
                    vivid_reddish_orange=(241, 58, 19),
                    dark_olive_green=(35, 44, 22))

    def csv_line_to_vec(self, line):
        line_list = line.strip().replace("\n", "").split(",")
        float_list = []
        for i in line_list:
            float_list.append(float(i))
        float_list = np.array(float_list)
        return float_list
