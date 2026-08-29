import cv2
import numpy as np


def apply_blur(image, severity="mild"):
    """
    Apply Gaussian blur to an image.

    Parameters
    ----------
    image : numpy.ndarray
        Input image in OpenCV format (BGR).

    severity : str
        Blur severity:
        - mild
        - moderate
        - severe

    Returns
    -------
    numpy.ndarray
        Blurred image.
    """

    kernel_sizes = {
        "mild": 5,
        "moderate": 11,
        "severe": 21
    }

    if severity not in kernel_sizes:
        raise ValueError(
            f"Unknown severity: {severity}. "
            f"Choose from {list(kernel_sizes.keys())}"
        )

    kernel_size = kernel_sizes[severity]

    blurred = cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        0
    )

    return blurred
def apply_underexposure(image, severity="mild"):
    """
    Simulate underexposure by reducing pixel intensity.

    Parameters
    ----------
    image : numpy.ndarray
        Input image in OpenCV BGR format.

    severity : str
        - mild
        - moderate
        - severe

    Returns
    -------
    numpy.ndarray
        Darkened image.
    """

    exposure_factors = {
        "mild": 0.70,
        "moderate": 0.45,
        "severe": 0.25
    }

    if severity not in exposure_factors:
        raise ValueError(
            f"Unknown severity: {severity}. "
            f"Choose from {list(exposure_factors.keys())}"
        )

    factor = exposure_factors[severity]

    # Convert to float so multiplication doesn't overflow.
    darkened = image.astype(np.float32) * factor

    # Keep valid image range.
    darkened = np.clip(
        darkened,
        0,
        255
    )

    return darkened.astype(np.uint8)
def apply_overexposure(image, severity="mild"):
    """
    Simulate overexposure by increasing pixel intensity.

    Parameters
    ----------
    image : numpy.ndarray
        Input image in OpenCV BGR format.

    severity : str
        - mild
        - moderate
        - severe

    Returns
    -------
    numpy.ndarray
        Overexposed image.
    """

    exposure_factors = {
        "mild": 1.20,
        "moderate": 1.50,
        "severe": 1.80
    }

    if severity not in exposure_factors:
        raise ValueError(
            f"Unknown severity: {severity}. "
            f"Choose from {list(exposure_factors.keys())}"
        )

    factor = exposure_factors[severity]

    brightened = image.astype(np.float32) * factor

    brightened = np.clip(
        brightened,
        0,
        255
    )

    return brightened.astype(np.uint8)
def apply_gaussian_noise(image, severity="mild"):
    """
    Add Gaussian noise to an image.

    Parameters
    ----------
    image : numpy.ndarray
        Input image in OpenCV BGR format.

    severity : str
        - mild
        - moderate
        - severe

    Returns
    -------
    numpy.ndarray
        Noisy image.
    """

    noise_std = {
        "mild": 8,
        "moderate": 20,
        "severe": 40
    }

    if severity not in noise_std:
        raise ValueError(
            f"Unknown severity: {severity}. "
            f"Choose from {list(noise_std.keys())}"
        )

    sigma = noise_std[severity]

    # Generate random Gaussian noise
    noise = np.random.normal(
        loc=0,
        scale=sigma,
        size=image.shape
    )

    # Add noise
    noisy = image.astype(np.float32) + noise

    # Keep pixels within valid range
    noisy = np.clip(
        noisy,
        0,
        255
    )

    return noisy.astype(np.uint8)


def apply_salt_pepper_noise(image, severity="mild"):
    """
    Add salt-and-pepper noise.

    Parameters
    ----------
    image : numpy.ndarray
        Input image in OpenCV BGR format.

    severity : str
        - mild
        - moderate
        - severe

    Returns
    -------
    numpy.ndarray
        Noisy image.
    """

    noise_probability = {
        "mild": 0.01,
        "moderate": 0.03,
        "severe": 0.06
    }

    if severity not in noise_probability:
        raise ValueError(
            f"Unknown severity: {severity}. "
            f"Choose from {list(noise_probability.keys())}"
        )

    probability = noise_probability[severity]

    noisy = image.copy()

    random_values = np.random.random(
        image.shape[:2]
    )

    # Pepper: set pixel to black
    pepper_mask = random_values < (
        probability / 2
    )

    noisy[pepper_mask] = 0

    # Salt: set pixel to white
    salt_mask = random_values > (
        1 - probability / 2
    )

    noisy[salt_mask] = 255

    return noisy
def apply_jpeg_corruption(image, severity="mild"):
    """
    Simulate JPEG compression artifacts.

    Parameters
    ----------
    image : numpy.ndarray
        Input image in OpenCV BGR format.

    severity : str
        - mild
        - moderate
        - severe

    Returns
    -------
    numpy.ndarray
        JPEG-compressed and decoded image.
    """

    jpeg_quality = {
        "mild": 40,
        "moderate": 20,
        "severe": 5
    }

    if severity not in jpeg_quality:
        raise ValueError(
            f"Unknown severity: {severity}. "
            f"Choose from {list(jpeg_quality.keys())}"
        )

    quality = jpeg_quality[severity]

    # Encode image as low-quality JPEG in memory
    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            quality
        ]
    )

    if not success:
        raise RuntimeError(
            "JPEG encoding failed."
        )

    # Decode it back into an image
    corrupted = cv2.imdecode(
        encoded,
        cv2.IMREAD_COLOR
    )

    if corrupted is None:
        raise RuntimeError(
            "JPEG decoding failed."
        )

    return corrupted