import argparse
import json

from PIL import Image
import torch

from .data import ImageTransform, IMAGENET_MEAN, IMAGENET_STD
from .evaluate import load_checkpoint_model


def predict(checkpoint, image_path, top_k=5, device=None):
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    model, config, device, temperature = load_checkpoint_model(checkpoint, device)
    data = config.get("data", {})
    transform = ImageTransform(False, image_size=data.get("image_size", 224),
                               resize_size=data.get("resize_size", 256),
                               mean=data.get("mean", IMAGENET_MEAN),
                               std=data.get("std", IMAGENET_STD))
    with Image.open(image_path) as image:
        view = transform(image)
    with torch.inference_mode():
        logits = model(transform.normalize(view)[None].to(device), temperature=temperature)
        probabilities, classes = logits.softmax(-1).topk(min(top_k, logits.shape[-1]))
    return [{"class_id": label, "probability": probability}
            for label, probability in zip(classes[0].tolist(), probabilities[0].tolist())]


def main():
    parser = argparse.ArgumentParser(description="Classify an image with a saved checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device")
    args = parser.parse_args()
    print(json.dumps(predict(args.checkpoint, args.image, args.top_k, args.device), indent=2))


if __name__ == "__main__":
    main()
