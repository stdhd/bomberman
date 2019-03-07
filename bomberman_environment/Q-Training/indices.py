import numpy as np

def x_y_to_index(x, y, ncols, nrows):
    """
    Return the index of a free grid from x, y coordinates. Indices start at 1 !

    Indexing starts at x, y = 0, 0 and increases row by row. Higher y value => Higher index

    Raises ValueError for wall coordinates!
    :param x: x coordinate
    :param y: y coordinate
    :return: Index of square x, y coords point to
    """

    if x >= ncols - 1 or x <= 0 or y >= nrows - 1 or y <= 0:
        raise ValueError("Coordinates outside of game grid")
    if (x + 1) * (y + 1) % 2 == 1:
        raise ValueError("Received wall coordinates!")

    else:
        # find full and half full rows below x, y coords
        ind = 0
        full_rows = y // 2
        half_rows = (y - 1) // 2

        ind += full_rows * (ncols - 2)
        ind += half_rows * np.ceil((ncols - 2) / 2)

        ind += (x + 1) // 2 if (full_rows + half_rows) % 2 == 1 else x

        return int(ind)


def index_to_x_y(ind, ncols, nrows):
    """
    Convert a given coordinate index into its x, y representation. Indices start at 1 !
    :param ind: Index of coordinate to represent as x, y
    :param ncols: Number of
    :param nrows:
    :return: x, y coordinates
    """

    y = 1

    full = True

    while ind > ncols - 2 and full == True:
        ind -= ncols - 2
        y += 1
        full = False

        while ind > np.ceil((ncols - 2) / 2) and full == False:
            ind -= np.ceil((ncols - 2) / 2)
            y += 1
            full = True

    if full:
        x = ind

    else:
        x = 2 * ind - 1

    return int(x), int(y)