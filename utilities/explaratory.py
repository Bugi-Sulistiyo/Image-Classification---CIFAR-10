import tensorflow_datasets as tfds
from collections import Counter
import matplotlib.pyplot as plt
from typing import  Literal

def count_labels(dataset, max_items=None):
    counter, i = Counter(), 0
    for _, label in tfds.as_numpy(dataset):
        counter[int(label)] += 1
        i += 1
        if max_items and i >= max_items:
            break
    return counter

def show_examples(dataset, class_name:str, num_sample:Literal[1, 3, 6, 9]=3):
    plt.figure(figsize=(6,6))

    if num_sample == 1:
        rows, cols = 1, 1
    elif num_sample == 3:
        rows, cols = 1, 3
    elif num_sample == 6:
        rows, cols = 2, 3
    else:
        rows, cols = 3, 3

    i=1
    for image, label in dataset.take(num_sample):
        plt.subplot(rows,cols, i)
        plt.imshow(image.numpy())
        plt.title(class_name[int(label.numpy())])
        plt.axis("off")
        i += 1
    plt.tight_layout()
    plt.show()