import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 6


# ============================================================
# LOAD PRETRAINED RESNET-18
# ============================================================

weights = ResNet18_Weights.DEFAULT

model = resnet18(weights=weights)


# ============================================================
# INSPECT ORIGINAL CLASSIFIER
# ============================================================

print("=" * 60)
print("ORIGINAL RESNET-18")
print("=" * 60)

print(model.fc)


# ============================================================
# REPLACE FINAL CLASSIFIER
# ============================================================

input_features = model.fc.in_features

model.fc = nn.Linear(
    input_features,
    NUM_CLASSES,
)


# ============================================================
# MODEL INFORMATION
# ============================================================

print()
print("=" * 60)
print("MODIFIED RESNET-18")
print("=" * 60)

print(model.fc)

print()
print(f"Input features : {input_features}")
print(f"Output classes : {NUM_CLASSES}")


# ============================================================
# TEST FORWARD PASS
# ============================================================

dummy_input = torch.randn(
    2,
    3,
    224,
    224,
)

output = model(dummy_input)


print()
print("=" * 60)
print("FORWARD PASS TEST")
print("=" * 60)

print(f"Input shape  : {dummy_input.shape}")
print(f"Output shape : {output.shape}")


# ============================================================
# VALIDATION
# ============================================================

assert output.shape == (
    2,
    NUM_CLASSES,
)


print()
print("=" * 60)
print("RESNET TEST PASSED")
print("=" * 60)