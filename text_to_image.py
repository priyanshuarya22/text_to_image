from collections.abc import Callable
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


class TextToImage:
    pipe: StableDiffusionPipeline | None = None

    def load_model(self) -> None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        pipe.to(device)
        self.pipe = pipe


    def generate(self, prompt: str, *, negative_prompt: str | None = None, num_steps: int = 50, callback: Callable[[int, int, torch.FloatTensor], None] | None = None, callback_steps: int = 1) -> Image.Image:
        if not self.pipe:
            raise RuntimeError("Pipeline is not loaded")
        return self.pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=num_steps, guidance_scale=9.0, callback=callback, callback_steps=callback_steps).images[0]
