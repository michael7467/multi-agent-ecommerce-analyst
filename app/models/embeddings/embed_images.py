from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from transformers import CLIPModel, CLIPProcessor


MODEL_NAME = "openai/clip-vit-base-patch32"
IMAGE_DIR = Path("data/raw/product_images")
METADATA_CSV = Path("data/processed/electronics_image_urls.csv")

EMBEDDINGS_OUTPUT = Path("artifacts/embeddings/image_embeddings.npy")
METADATA_OUTPUT = Path("artifacts/embeddings/image_embedding_metadata.csv")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
DEFAULT_BATCH_SIZE = 32


@dataclass(frozen=True)
class ImageRecord:
    product_id: str
    title: str
    image_url: str
    image_path: Path


class ImageEmbedder:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size

        print(f"Loading model: {model_name}")
        print(f"Using device: {self.device}")

        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        print(f"Loaded model class: {self.model.__class__.__name__}")

    @staticmethod
    def _load_image(image_path: Path) -> Image.Image:
        try:
            return Image.open(image_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"Could not load image '{image_path}': {exc}") from exc

    @staticmethod
    def _normalize(features: torch.Tensor) -> torch.Tensor:
        return features / features.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)

    def _extract_features(self, output) -> torch.Tensor:
        """
        Robustly convert different model return types into a tensor of features.
        Handles:
        - torch.Tensor
        - CLIP-style outputs with .image_embeds
        - vision outputs with .pooler_output
        - tuple/list outputs
        """
        if isinstance(output, torch.Tensor):
            return output

        if hasattr(output, "image_embeds") and output.image_embeds is not None:
            return output.image_embeds

        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output

        if isinstance(output, (tuple, list)) and len(output) > 0:
            first = output[0]
            if isinstance(first, torch.Tensor):
                return first

        raise TypeError(
            "Could not extract image features from model output. "
            f"Got type: {type(output)}"
        )

    def _forward_image_features(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Try the standard CLIP path first. If the environment returns something unexpected,
        fall back to the full forward pass and then to the vision model.
        """
        with torch.no_grad():
            # Preferred official CLIP path
            try:
                output = self.model.get_image_features(**inputs)
                features = self._extract_features(output)
                return self._normalize(features)
            except Exception as exc:
                print(f"get_image_features failed, trying model(**inputs): {exc}")

            # Fallback: full CLIP forward
            try:
                output = self.model(**inputs)
                features = self._extract_features(output)
                return self._normalize(features)
            except Exception as exc:
                print(f"model(**inputs) failed, trying vision_model(**inputs): {exc}")

            # Final fallback: vision backbone output
            if hasattr(self.model, "vision_model"):
                output = self.model.vision_model(pixel_values=inputs["pixel_values"])
                features = self._extract_features(output)
                return self._normalize(features)

        raise RuntimeError("All embedding strategies failed.")

    def embed_batch(self, image_paths: list[Path]) -> np.ndarray:
        images = [self._load_image(path) for path in image_paths]

        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        image_features = self._forward_image_features(inputs)
        return image_features.cpu().numpy().astype(np.float32)

    def embed_images(self, image_paths: list[Path]) -> tuple[np.ndarray, list[int]]:
        all_embeddings: list[np.ndarray] = []
        success_indices: list[int] = []

        total = len(image_paths)
        if total == 0:
            raise ValueError("No images provided for embedding.")

        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            batch_paths = image_paths[start:end]

            valid_paths: list[Path] = []
            valid_indices: list[int] = []

            for local_idx, image_path in enumerate(batch_paths):
                global_idx = start + local_idx
                try:
                    # validate loading early
                    self._load_image(image_path)
                    valid_paths.append(image_path)
                    valid_indices.append(global_idx)
                except Exception as exc:
                    print(f"Failed to load {image_path.name}: {exc}")

            if not valid_paths:
                print(f"Skipped batch {start}:{end} because no images were valid.")
                continue

            try:
                batch_embeddings = self.embed_batch(valid_paths)
                all_embeddings.append(batch_embeddings)
                success_indices.extend(valid_indices)

            except Exception as exc:
                print(
                    f"Batch {start}:{end} failed during embedding. "
                    f"Falling back to single-image mode. Error: {exc}"
                )

                for image_path, global_idx in zip(valid_paths, valid_indices):
                    try:
                        emb = self.embed_batch([image_path])[0]
                        all_embeddings.append(np.expand_dims(emb, axis=0))
                        success_indices.append(global_idx)
                    except Exception as single_exc:
                        print(f"Failed to embed {image_path.name}: {single_exc}")

            print(f"Processed {end}/{total} images")

        if not all_embeddings:
            raise ValueError("No image embeddings were created.")

        embeddings = np.vstack(all_embeddings)
        return embeddings, success_indices


def find_local_image(product_id: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = IMAGE_DIR / f"{product_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def build_image_metadata() -> pd.DataFrame:
    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Missing metadata CSV: {METADATA_CSV}")

    df = pd.read_csv(METADATA_CSV)

    if "product_id" not in df.columns:
        raise ValueError("Metadata CSV must contain a 'product_id' column.")

    rows: list[dict[str, str]] = []

    for _, row in df.iterrows():
        product_id = str(row["product_id"]).strip()
        if not product_id:
            continue

        image_path = find_local_image(product_id)
        if image_path is None:
            continue

        rows.append(
            {
                "product_id": product_id,
                "title": str(row.get("title", "")),
                "image_url": str(row.get("image_url", "")),
                "image_path": str(image_path),
            }
        )

    metadata_df = pd.DataFrame(rows)

    if metadata_df.empty:
        raise ValueError("No local images found to embed.")

    return metadata_df


def save_image_embeddings(batch_size: int = DEFAULT_BATCH_SIZE) -> None:
    metadata_df = build_image_metadata()
    image_paths = [Path(p) for p in metadata_df["image_path"].tolist()]

    print(f"Found {len(image_paths)} local images to process")

    embedder = ImageEmbedder(batch_size=batch_size)
    embeddings, success_indices = embedder.embed_images(image_paths)

    metadata_df = metadata_df.iloc[success_indices].reset_index(drop=True)

    if len(metadata_df) != len(embeddings):
        raise RuntimeError(
            "Metadata and embeddings are misaligned after processing. "
            f"metadata rows={len(metadata_df)}, embeddings rows={len(embeddings)}"
        )

    EMBEDDINGS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_OUTPUT, embeddings.astype(np.float32))
    metadata_df.to_csv(METADATA_OUTPUT, index=False)

    print(f"Saved image embeddings to: {EMBEDDINGS_OUTPUT}")
    print(f"Saved image metadata to: {METADATA_OUTPUT}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(metadata_df.head())


if __name__ == "__main__":
    save_image_embeddings()