"""Production inference for COS30082 Model 8 (ResNet18 + Mixup).

Uses the exact deterministic preprocessing from src/dataset.py:
PIL RGB -> Resize((224, 224)) -> ToTensor -> ImageNet Normalize.
No training datasets or training-time Mixup are needed for inference.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18


APP_DIR = Path(__file__).resolve().parent
MODEL_FILENAME = "model8_mixup_best_accuracy.pt"
CLASS_FILENAME = "class_names.json"
NUM_CLASSES = 200
SELECTED_EPOCH = 18
IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def find_resource(filename: str, local_relative_path: str) -> Path:
    """Support both local project layout and standalone Hugging Face layout."""
    choices = [APP_DIR / filename, APP_DIR.parent / local_relative_path]
    for path in choices:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Missing {filename}. Tried: {', '.join(map(str, choices))}. "
        "For local use, keep the model in models/ and labels in data/splits/. "
        "For Spaces, use scripts/prepare_space.py to collect the deployment files."
    )


def load_species_names(path: Path) -> list[str]:
    """Class IDs must match the exact 0..199 order used in training."""
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict) or set(raw) != {str(i) for i in range(NUM_CLASSES)}:
        raise ValueError("class_names.json must map every label 0..199 to a species name")
    names = [raw[str(i)] for i in range(NUM_CLASSES)]
    if any(not isinstance(s, str) or not s.strip() for s in names):
        raise ValueError("class_names.json contains missing/invalid species names")
    if len(set(names)) != NUM_CLASSES:
        raise ValueError("class_names.json contains duplicate species names")
    return names


class BirdPredictor:
    def __init__(self, device: str | None = None):
        model_path = find_resource(MODEL_FILENAME, f"models/{MODEL_FILENAME}")
        names_path = find_resource(CLASS_FILENAME, f"data/splits/{CLASS_FILENAME}")
        self.names = load_species_names(names_path)

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # Only load data from the project's own trusted checkpoint file.
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError("Checkpoint does not contain model_state_dict")
        if (checkpoint.get("architecture_id") or checkpoint.get("model_name")) != "resnet18":
            raise ValueError("Expected an ImageNet-pretrained ResNet18 checkpoint")
        if int(checkpoint.get("epoch", -1)) != SELECTED_EPOCH:
            raise ValueError("Expected the selected Model 8 checkpoint from epoch 18")
        if int(checkpoint.get("image_size", -1)) != IMAGE_SIZE:
            raise ValueError("Model 8 requires input resolution 224x224")
        if int(checkpoint.get("num_classes", -1)) != NUM_CLASSES:
            raise ValueError("Model 8 must output predictions for 200 classes")
        if checkpoint.get("eval_preprocessing") != "dataset_clean_v1":
            raise ValueError("Checkpoint preprocessing metadata does not match the evaluator")
        if checkpoint.get("mixup_alpha") is None:
            raise ValueError("Checkpoint has no Model 8 Mixup metadata")

        self.model = resnet18(weights=None, num_classes=NUM_CLASSES)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model = self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
        self.model_path = model_path
        self.selected_epoch = int(checkpoint["epoch"])

    @torch.inference_mode()
    def predict_top5(self, image: Image.Image) -> list[tuple[str, float]]:
        """Return five species and full-softmax probabilities in descending order."""
        if image is None:
            raise ValueError("Upload an image before identifying a bird")
        if not isinstance(image, Image.Image):
            raise TypeError("Expected a PIL image")
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        if logits.shape != (1, NUM_CLASSES) or not torch.isfinite(logits).all():
            raise RuntimeError("The classifier produced an invalid output")
        probs = torch.softmax(logits.float(), dim=1)[0]
        top_probs, top_ids = torch.topk(probs, k=5)
        return [(self.names[int(idx)], float(score))
                for idx, score in zip(top_ids.tolist(), top_probs.tolist())]
