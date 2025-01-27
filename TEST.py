import numpy as np

def process_list(input_list):
    input_list = np.array(input_list, dtype=float)
    if input_list.ndim == 1:
        input_list = input_list.reshape(-1, 1)
    input_list = input_list ** 2 + np.sum(input_list)
    return input_list
