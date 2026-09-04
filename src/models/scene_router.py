from __future__ import annotations

from collections import Counter

import numpy as np
from PIL import Image

SCENE_LABELS = ("face", "text", "plant", "clock", "bird")

# Lightweight keyword / color heuristics so the project runs without CLIP.
_SCENE_HINTS = {
    "face": ("face", "person", "portrait", "people", "人脸", "人像"),
    "text": ("text", "ocr", "sign", "word", "文字", "字幕"),
    "plant": ("plant", "leaf", "tree", "green", "绿植", "植被"),
    "clock": ("clock", "watch", "time", "钟表", "手表"),
    "bird": ("bird", "feather", "鸟"),
}


class SceneRouter:
    def __init__(self, cfg: dict | None = None) -> None:
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.backend = str(cfg.get("backend", "heuristic"))
        self.min_confidence = float(cfg.get("min_confidence", 0.35))
        self.clip_model = cfg.get("clip_model", "ViT-B-32")
        self.clip_pretrained = cfg.get("clip_pretrained", "openai")
        self.overrides = {str(k): str(v) for k, v in (cfg.get("overrides") or {}).items()}
        self._clip = None

    def infer(self, image: Image.Image, name: str = "") -> str:
        if not self.enabled:
            return "general"
        if name in self.overrides:
            return self.overrides[name]

        by_name = self._from_name(name)
        if by_name:
            return by_name

        if self.backend == "clip":
            scene, score = self._from_clip(image)
            if score >= self.min_confidence:
                return scene
        return self._from_color(image)

    def _from_name(self, name: str) -> str | None:
        lowered = name.lower()
        for scene, hints in _SCENE_HINTS.items():
            if any(hint.lower() in lowered for hint in hints):
                return scene
        return None

    def _from_color(self, image: Image.Image) -> str:
        """Very coarse fallback: green-dominant images go to plant."""
        thumb = np.asarray(image.convert("RGB").resize((64, 64)), dtype=np.int16)
        r, g, b = thumb[..., 0], thumb[..., 1], thumb[..., 2]
        green_ratio = float(np.mean((g > r + 15) & (g > b + 10)))
        if green_ratio > 0.35:
            return "plant"
        return "general"

    def _from_clip(self, image: Image.Image) -> tuple[str, float]:
        model, preprocess, tokenizer = self._load_clip()
        import torch

        prompts = [
            "a photo of a small human face",
            "a photo containing readable text or letters",
            "a photo of dense plants or foliage",
            "a photo of a clock or wristwatch",
            "a photo of a bird",
        ]
        image_t = preprocess(image).unsqueeze(0)
        text_t = tokenizer(prompts)
        with torch.no_grad():
            image_f = model.encode_image(image_t)
            text_f = model.encode_text(text_t)
            image_f = image_f / image_f.norm(dim=-1, keepdim=True)
            text_f = text_f / text_f.norm(dim=-1, keepdim=True)
            probs = (100.0 * image_f @ text_f.T).softmax(dim=-1)[0]
        idx = int(probs.argmax().item())
        return SCENE_LABELS[idx], float(probs[idx].item())

    def _load_clip(self):
        if self._clip is not None:
            return self._clip
        try:
            import open_clip
        except ImportError as exc:
            raise RuntimeError("CLIP backend requires open-clip-torch. Use backend=heuristic.") from exc
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.clip_model, pretrained=self.clip_pretrained
        )
        tokenizer = open_clip.get_tokenizer(self.clip_model)
        model.eval()
        self._clip = (model, preprocess, tokenizer)
        return self._clip


def majority_scene(scenes: list[str]) -> str:
    if not scenes:
        return "general"
    return Counter(scenes).most_common(1)[0][0]
