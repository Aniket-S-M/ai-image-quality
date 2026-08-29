import cv2
import numpy as np


def calculate_sharpness(image):
    """
    Calculate image sharpness using
    variance of Laplacian.

    Higher value generally indicates
    more high-frequency detail.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    return float(laplacian.var())
def calculate_brightness(image):
    """
    Calculate average grayscale intensity.

    Higher value -> brighter image
    Lower value  -> darker image
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return float(gray.mean())
def calculate_highlight_clipping(image):
    """
    Calculate the percentage of pixels
    near maximum intensity.

    Higher value -> more highlight clipping.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    clipped_pixels = (
        gray >= 250
    ).sum()

    total_pixels = gray.size

    percentage = (
        clipped_pixels /
        total_pixels
    ) * 100

    return float(percentage)
def calculate_contrast(image):
    """
    Calculate image contrast using the standard
    deviation of grayscale intensity.

    Higher value -> greater intensity variation.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return float(gray.std())
def calculate_saturation(image):
    """
    Calculate average color saturation.

    Higher value -> more saturated colors.
    Lower value  -> less saturated colors.
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    saturation = hsv[:, :, 1]

    return float(saturation.mean())
def calculate_edge_density(image):
    """
    Calculate the proportion of pixels detected
    as edges using Canny edge detection.

    Higher value -> more detected edges.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    edges = cv2.Canny(
        gray,
        100,
        200
    )

    edge_pixels = np.count_nonzero(edges)

    total_pixels = edges.size

    return float(
        edge_pixels / total_pixels
    )
def calculate_noise_estimate(image):
    """
    Estimate high-frequency variation using
    the residual between the original grayscale
    image and a Gaussian-smoothed version.

    Higher value generally indicates more
    high-frequency variation.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    smooth = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    residual = (
        gray.astype(np.float32)
        - smooth.astype(np.float32)
    )

    return float(residual.std())
