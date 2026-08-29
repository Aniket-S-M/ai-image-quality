from pathlib import Path

import torch
from torchvision import transforms

from image_quality_dataset import ImageQualityDataset


TRAIN_DIR = Path(
    "data/generated/dataset_expanded/train"
)

IMAGE_SIZE = 224


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


dataset = ImageQualityDataset(
    TRAIN_DIR,
    transform=transform,
)


print("=" * 60)
print("IMAGE QUALITY DATASET TEST")
print("=" * 60)

print(f"Dataset size: {len(dataset)}")

print()
print("Classes:")

for name, idx in dataset.CLASS_TO_IDX.items():
    print(f"  {idx}: {name}")


print()
print("=" * 60)
print("SAMPLE CHECK")
print("=" * 60)

for index in [0, 1, 2, 3, 4]:

    image, label = dataset[index]

    filename = dataset.image_paths[index].name

    print()
    print(f"Sample {index}")
    print(f"Filename    : {filename}")
    print(f"Image shape : {image.shape}")
    print(f"Image type  : {image.dtype}")
    print(f"Label       : {label}")
    print(
        f"Class       : {dataset.CLASS_NAMES[label]}"
    )


image, label = dataset[0]

assert isinstance(image, torch.Tensor)
assert image.shape == (3, 224, 224)
assert image.dtype == torch.float32
assert 0 <= label < len(dataset.CLASS_NAMES)


print()
print("=" * 60)
print("DATASET TEST PASSED")
print("=" * 60)