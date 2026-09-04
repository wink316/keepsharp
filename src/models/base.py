from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from PIL import Image


@dataclass
class EnhanceContext:
    name: str
    scene: str = "general"
    prompt: str = ""
    negative_prompt: str = ""
    strength: float = 0.25
    guidance_scale: float = 3.8
    extra: dict = field(default_factory=dict)


class BaseEnhancer(ABC):
    name = "base"

    @abstractmethod
    def enhance(self, image: Image.Image, context: EnhanceContext) -> Image.Image:
        raise NotImplementedError
