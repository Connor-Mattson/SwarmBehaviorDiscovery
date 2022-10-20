import numpy as np


class DataAggregationArchive:
    def __init__(self, file=None, scalar=False):
        self.archive = np.array([[]])
        self.scalar = scalar
        if scalar:
            self.archive = np.array([])
        if file is not None:
            self.archive = np.array([])
            self.read_from_file(file)

    def append_scalars(self, scalars):
        self.archive = np.append(self.archive, scalars)

    def append(self, anchor, pos, neg):
        if [anchor, pos, neg] in self.archive.tolist() or [pos, anchor, neg] in self.archive.tolist():
            return
        if len(self.archive[0]) == 0:
            self.archive = np.array([[anchor, pos, neg]])
        else:
            self.archive = np.concatenate((self.archive, np.array([[anchor, pos, neg]])))

    def __len__(self):
        return len(self.archive)

    def __getitem__(self, item):
        return self.archive[item][0], self.archive[item][1], self.archive[item][2]

    def read_from_file(self, file_name):
        f = open(file_name, "r")
        lines = f.readlines()
        for line in lines:
            if self.scalar:
                self.archive = np.append(self.archive, int(line))
                continue
            line_list = line.split(",")
            triplet = []
            for i in line_list:
                triplet.append(int(i))
            float_list = np.array([triplet])
            if len(self.archive) == 0:
                self.archive = float_list
            else:
                self.archive = np.concatenate((self.archive, float_list))

    def save_to_file(self, file_name):
        f = open(file_name, "w")
        for value in self.archive:
            line = ""
            if not self.scalar:
                for i, val in enumerate(value):
                    line += str(val)
                    if i < len(value) - 1:
                        line += ", "
            else:
                line += str(value)
            line += "\n"
            f.write(line)
        f.close()