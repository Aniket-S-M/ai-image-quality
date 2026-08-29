from pathlib import Path

import cv2

from features import (
    calculate_sharpness,
    calculate_brightness,
    calculate_highlight_clipping,
    calculate_contrast,
    calculate_saturation,
    calculate_edge_density,
    calculate_noise_estimate
    
)

ROOT = Path(__file__).resolve().parents[1]

CLEAN_DIR = ROOT / "data" / "raw" / "clean"
BLUR_DIR = ROOT / "data" / "generated" / "blur_test"


# Original image
image_path = sorted(CLEAN_DIR.glob("*.jpg"))[0]

image = cv2.imread(str(image_path))

if image is None:
    raise RuntimeError(f"Could not read {image_path}")

print("Sharpness measurements:\n")
print("\nBrightness measurements:\n")

print(
    f"Original  : "
    f"{calculate_brightness(image):.2f}"
)

for severity in [
    "mild",
    "moderate",
    "severe"
]:

    under_path = (
        ROOT
        / "data"
        / "generated"
        / "underexposure_test"
        / f"{image_path.stem}_underexposure_{severity}.jpg"
    )

    underexposed = cv2.imread(
        str(under_path)
    )

    if underexposed is None:
        raise RuntimeError(
            f"Could not read {under_path}"
        )

    print(
        f"{severity.capitalize():9s}: "
        f"{calculate_brightness(underexposed):.2f}"
    )


print(
    f"Original  : "
    f"{calculate_sharpness(image):.2f}"
)


# Generated blur images
for severity in ["mild", "moderate", "severe"]:

    blur_path = (
        BLUR_DIR
        / f"{image_path.stem}_blur_{severity}.jpg"
    )

    blurred = cv2.imread(
        str(blur_path)
    )

    if blurred is None:
        raise RuntimeError(
            f"Could not read {blur_path}"
        )

    print(
        f"{severity.capitalize():9s}: "
        f"{calculate_sharpness(blurred):.2f}"
    )
print("\nHighlight clipping measurements:\n")

print(
    f"Original  : "
    f"{calculate_highlight_clipping(image):.2f}%"
)

for severity in [
    "mild",
    "moderate",
    "severe"
]:

    over_path = (
        ROOT
        / "data"
        / "generated"
        / "overexposure_test"
        / f"{image_path.stem}_overexposure_{severity}.jpg"
    )

    overexposed = cv2.imread(
        str(over_path)
    )

    if overexposed is None:
        raise RuntimeError(
            f"Could not read {over_path}"
        )

    print(
        f"{severity.capitalize():9s}: "
        f"{calculate_highlight_clipping(overexposed):.2f}%"
    )
print("\nContrast measurements:\n")

print(
    f"Original: "
    f"{calculate_contrast(image):.2f}"
)
print("\nSaturation measurement:\n")

print(
    f"Original: "
    f"{calculate_saturation(image):.2f}"
)
print("\nEdge density measurement:\n")

print(
    f"Original: "
    f"{calculate_edge_density(image):.4f}"
)
print("\nNoise estimate measurements:\n")

print(
    f"Original: "
    f"{calculate_noise_estimate(image):.2f}"
)

for severity in [
    "mild",
    "moderate",
    "severe"
]:

    noise_path = (
        ROOT
        / "data"
        / "generated"
        / "noise_test"
        / f"{image_path.stem}_noise_{severity}.jpg"
    )

    noisy = cv2.imread(
        str(noise_path)
    )

    if noisy is None:
        raise RuntimeError(
            f"Could not read {noise_path}"
        )

    print(
        f"{severity.capitalize():9s}: "
        f"{calculate_noise_estimate(noisy):.2f}"
    )