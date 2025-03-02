import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to display one image
def display_image(img, title='', size=None, show_axis=False):
    plt.gray()
    if not show_axis:
        plt.axis('off')
    h = plt.imshow(img, interpolation='none')
    if size:
        dpi = plt.gcf().get_dpi() / size  # Use plt.gcf() to get the current figure
        plt.gcf().set_figwidth(img.shape[1] / dpi)
        plt.gcf().set_figheight(img.shape[0] / dpi)
        plt.gca().set_position([0, 0, 1, 1])  # Use plt.gca() to get the current axes
        if show_axis:
            plt.gca().set_xlim(-1, img.shape[1])
            plt.gca().set_ylim(img.shape[0], -1)
    plt.grid(False)
    plt.title(title)
    plt.show()


