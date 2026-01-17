import torch
from diffusers import StableDiffusionPipeline
import os
from tqdm import tqdm

MODEL_ID = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16
)
pipe.to("cuda")
pipe.enable_attention_slicing()

dog_breeds = [
    "Labrador Retriever",
    "German Shepherd",
    "Golden Retriever",
    "Bulldog",
    "Poodle",
    "Beagle"
]

IMAGES_PER_BREED = 150

os.makedirs("dog_dataset", exist_ok=True)

for breed in dog_breeds:
    breed_dir = f"dog_dataset/{breed.replace(' ', '_')}"
    os.makedirs(breed_dir, exist_ok=True)

    for i in tqdm(range(IMAGES_PER_BREED), desc=f"Generating {breed}"):
        prompt = f"a realistic high quality photo of a {breed}, natural lighting, sharp focus"
        image = pipe(prompt).images[0]
        image.save(f"{breed_dir}/{i}.png")
