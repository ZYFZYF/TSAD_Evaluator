# @Time    : 2022/3/16 15:24
# @Author  : ZYF
import numpy as np


def sliding(data: np.ndarray, window_size: int):
    for i in range(data.shape[0] - window_size + 1):
        yield data[i:i + window_size]


if __name__ == '__main__':
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    for y in sliding(x, 2):
        print(y)
    z = list(sliding(x, 2))
    print(type(z[0]))
