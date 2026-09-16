import numpy as np


def addition(x1, x2):
    return x1 + x2


def square_plus(x1, x2):
    return x1 * x1 + x2


def product_sum(x1, x2):
    return x1 * (x1 + x2)


def difference_squares(x1, x2):
    return (x1 - x2) * (x1 + x2)


def cubic(x1):
    return x1 * x1 * x1 + x1 * x1 + x1


def sine(x1):
    return np.sin(x1)


def polynomial_sine(x1):
    return x1 * x1 + x1 + np.sin(x1)


def nguyen_1(x1):
    return x1 + x1**2 + x1**3


def nguyen_2(x1):
    return x1 + x1**2 + x1**3 + x1**4


def nguyen_3(x1):
    return x1 + x1**2 + x1**3 + x1**4 + x1**5


def nguyen_4(x1):
    return x1 + x1**2 + x1**3 + x1**4 + x1**5 + x1**6


def nguyen_5(x1):
    return np.sin(x1**2) * np.cos(x1)


def nguyen_6(x1):
    return np.sin(x1) + np.sin(x1 + x1**2)


def nguyen_7(x1):
    return np.log(x1 + 1) + np.log(x1**2)


def nguyen_8(x1):
    return np.sqrt(x1)


def nguyen_9(x1, x2):
    return np.sin(x1) + np.sin(x2**2)


def nguyen_10(x1, x2):
    return np.sin(x1) * np.cos(x2)


def nguyen_12(x1, x2):
    return x1**4 - x1**3 + x2**2 - x2


TARGETS = {
    "addition": (addition, "x1 + x2"),
    "square_plus": (square_plus, "x1 * x1 + x2"),
    "product_sum": (product_sum, "x1 * (x1 + x2)"),
    "difference_squares": (difference_squares, "(x1 - x2) * (x1 + x2)"),
    "cubic": (cubic, "x1 * x1 * x1 + x1 * x1 + x1"),
    "sine": (sine, "sin(x1)"),
    "polynomial_sine": (polynomial_sine, "x1 * x1 + x1 + sin(x1)"),
    "nguyen_1": (nguyen_1, "x1 + x1**2 + x1**3"),
    "nguyen_2": (nguyen_2, "x1 + x1**2 + x1**3 + x1**4"),
    "nguyen_3": (nguyen_3, "x1 + x1**2 + x1**3 + x1**4 + x1**5"),
    "nguyen_4": (nguyen_4, "x1 + x1**2 + x1**3 + x1**4 + x1**5 + x1**6"),
    "nguyen_5": (nguyen_5, "sin(x1**2) * cos(x1)"),
    "nguyen_6": (nguyen_6, "sin(x1) + sin(x1 + x1**2)"),
    "nguyen_7": (nguyen_7, "log(x1 + 1) + log(x1**2)"),
    "nguyen_8": (nguyen_8, "sqrt(x1)"),
    "nguyen_9": (nguyen_9, "sin(x1) + sin(x2**2)"),
    "nguyen_10": (nguyen_10, "sin(x1) * cos(x2)"),
    "nguyen_12": (nguyen_12, "x1**4 - x1**3 + x2**2 - x2"),
}


TARGET_DOMAINS = {
    nguyen_1: (-5.0, 5.0),
    nguyen_2: (-5.0, 5.0),
    nguyen_3: (-5.0, 5.0),
    nguyen_4: (-5.0, 5.0),
    nguyen_5: (-5.0, 5.0),
    nguyen_6: (-5.0, 5.0),
    nguyen_7: (1e-6, 5.0),
    nguyen_8: (0.0, 5.0),
    nguyen_9: (-5.0, 5.0),
    nguyen_10: (-5.0, 5.0),
    nguyen_12: (-5.0, 5.0),
}
