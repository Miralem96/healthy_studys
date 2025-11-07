import numpy as np

def confidence_interval(data):
    mean = np.mean(data)
    margin = 1.96 * (np.std(data) / np.sqrt(len(data)))
    return mean - margin, mean + margin