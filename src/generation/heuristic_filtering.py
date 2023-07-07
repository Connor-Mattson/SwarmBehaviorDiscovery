"""
Filters out uninteresting homogeneous controllers.
"""

import numpy as np


class ControllerFilter:

    @staticmethod
    def _euclidean_distance(c1, c2):
        difference = np.array(c1) - np.array(c2)
        squared_difference = np.power(difference, 2)
        sum_squared_difference = np.sum(squared_difference)
        distance = np.sqrt(sum_squared_difference)
        return distance

    @staticmethod
    def _element_magnitude_test(controller):
        threshold = 0.4
        return max(-min(controller), max(controller)) < threshold

    @staticmethod
    def _controller_magnitude_test(controller):
        threshold = 0.65
        controller_vec = np.array(controller)
        l2_norm = np.linalg.norm(controller_vec)
        return l2_norm < threshold

    @staticmethod
    def _displacement_test(controller):
        threshold = 0.5
        a, b, c, d = controller
        return abs(a+b) + abs(c+d) < threshold

    @staticmethod
    def _mirror_distance_test(controller):
        threshold = 0.2
        a, b, = controller[0], controller[1]
        mirror = [a, b, -a, -b]
        return ControllerFilter._euclidean_distance(controller, mirror) < threshold

    @staticmethod
    def _neglectful_distance_test(controller):
        threshold = 0.3
        a, b, = controller[0], controller[1]
        neglectful_controller = [a, b, a, b]
        return ControllerFilter._euclidean_distance(controller, neglectful_controller) < threshold

    @staticmethod
    def homogeneous_filter(controller):
        element_magnitude = ControllerFilter._element_magnitude_test(controller)
        controller_magnitude = ControllerFilter._controller_magnitude_test(controller)
        displacement = ControllerFilter._displacement_test(controller)
        mirror_distance = ControllerFilter._mirror_distance_test(controller)
        neglectful_distance = ControllerFilter._neglectful_distance_test(controller)
        return element_magnitude \
            and controller_magnitude \
            and displacement \
            and mirror_distance \
            and neglectful_distance
