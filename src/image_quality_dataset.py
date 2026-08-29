from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class ImageQualityDataset(Dataset):

    CLASS_NAMES = [
        "blur",
        "corruption",
        "noise",
        "none",
        "overexposure",
        "underexposure",
    ]

    CLASS_TO_IDX = {
        name: idx
        for idx, name in enumerate(CLASS_NAMES)
    }

    def __init__(self, root_dir, transform=None):

        self.root_dir = Path(root_dir)
        self.transform = transform

        self.image_paths = sorted(
            self.root_dir.glob("*.jpg")
        )

        if not self.image_paths:
            raise RuntimeError(
                f"No JPG images found in {self.root_dir}"
            )

    def __len__(self):
        return len(self.image_paths)

    def _get_label_from_filename(self, filename):

        stem = Path(filename).stem
        parts = stem.split("_")

        # Example:
        # 000000000776_clean.jpg
        if parts[-1] == "clean":
            return self.CLASS_TO_IDX["none"]

        # Example:
        # 000000000776_blur_mild.jpg
        #
        # Example:
        # 000000000776_noise_sp_moderate.jpg
        #
        # In both cases, parts[1] is the degradation type.
        degradation = parts[1]

        if degradation not in self.CLASS_TO_IDX:
            raise ValueError(
                f"Unknown degradation '{degradation}' "
                f"in filename '{filename}'"
            )

        return self.CLASS_TO_IDX[degradation]

    def __getitem__(self, index):

        image_path = self.image_paths[index]

        image = Image.open(image_path).convert("RGB")

        label = self._get_label_from_filename(
            image_path.name
        )

        if self.transform is not None:
            image = self.transform(image)

        return image, label