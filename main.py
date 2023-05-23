import numpy as np
from simplex import simplex


def main():
    a = np.array([[-1, 1, -1, -1],
                  [2, 4, 0, 0],
                  [0, 0, 1, 1]])
    b = np.array([8, 10, 3])
    c = np.array([2, -3, 0, -5])
    simplex(a, b, c)

    b = np.array([2, 5, 0])
    simplex(a, b, c)


if __name__ == "__main__":
    main()