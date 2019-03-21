

import matplotlib.pyplot as plt
import json
import numpy as np


def main():
    """
    Load and shows a progress file.
    :return:
    """

    with open("progress.json", "r") as f:
        x, y = json.load(f)


    for i, val in enumerate(x):
        if i > 0:
            x[i] += x[i-1]

    x = np.array(x)

    x *= 32

    plt.plot(x, y)

    plt.show()

if __name__ == '__main__':
    main()

