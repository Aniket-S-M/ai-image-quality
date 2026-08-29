from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path(
    "best_resnet18_finetuned.pth"
)

IMAGE_SIZE = 224

CLASS_NAMES = [
    "blur",
    "corruption",
    "noise",
    "none",
    "overexposure",
    "underexposure",
]


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406,
        ],
        std=[
            0.229,
            0.224,
            0.225,
        ],
    ),
])


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    model = resnet18(
        weights=None
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        len(CLASS_NAMES),
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE,
        )
    )

    model = model.to(DEVICE)

    model.eval()

    return model


model = load_model()


# ============================================================
# PREDICTION
# ============================================================

def predict_image(image: Image.Image):

    image = image.convert("RGB")

    tensor = transform(image)

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(DEVICE)

    with torch.no_grad():

        outputs = model(tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )

        confidence, predicted_index = (
            probabilities.max(dim=1)
        )

    predicted_index = predicted_index.item()

    confidence = confidence.item()

    probabilities = probabilities[0].cpu().tolist()

    result = {
        "prediction": CLASS_NAMES[predicted_index],
        "confidence": confidence,
        "probabilities": {
            CLASS_NAMES[i]: probabilities[i]
            for i in range(len(CLASS_NAMES))
        },
    }

    return result