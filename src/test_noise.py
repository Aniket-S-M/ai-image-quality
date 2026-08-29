from pathlib import Path

import cv2

from degradation import (
    apply_gaussian_noise,
    apply_salt_pepper_noise
)


ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = ROOT / "data" / "raw" / "clean"

OUTPUT_DIR = (
    ROOT
    / "data"
    / "generated"
    / "noise_test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def calculate_mean_difference(original, noisy):
    """
    Measure average absolute pixel difference
    between original and noisy images.
    """

    difference = cv2.absdiff(
        original,
        noisy
    )

    return difference.mean()


# -----------------------------------------
# Load image
# -----------------------------------------

image_path = sorted(
    INPUT_DIR.glob("*.jpg")
)[0]

print("Input image:")
print(image_path)

image = cv2.imread(
    str(image_path)
)

if image is None:
    raise RuntimeError(
        f"Could not read image: {image_path}"
    )


# -----------------------------------------
# Generate noise
# -----------------------------------------

for severity in [
    "mild",
    "moderate",
    "severe"
]:

    noisy = apply_gaussian_noise(
        image,
        severity
    )

    output_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_noise_{severity}.jpg"
    )

    cv2.imwrite(
        str(output_path),
        noisy
    )

    difference = calculate_mean_difference(
        image,
        noisy
    )

    print(
        f"{severity.capitalize():10s} "
        f"mean pixel difference: "
        f"{difference:.2f}"
    )


print("\nGaussian noise test complete.")
print("\nTesting salt-and-pepper noise...\n")

for severity in [
    "mild",
    "moderate",
    "severe"
]:

    noisy = apply_salt_pepper_noise(
        image,
        severity
    )

    output_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_saltpepper_{severity}.jpg"
    )

    cv2.imwrite(
        str(output_path),
        noisy
    )

    difference = calculate_mean_difference(
        image,
        noisy
    )

    print(
        f"{severity.capitalize():10s} "
        f"mean pixel difference: "
        f"{difference:.2f}"
    )

print("\nNoise tests complete.")