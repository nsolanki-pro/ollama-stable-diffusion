from diffusers import DiffusionPipeline
import torch
import matplotlib.pyplot as plt

# Load pipeline with disabled safety checker
pipeline = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None  # Disable safety filter
)
pipeline.to("cuda")

# Set seed for reproducibility
seed = 1234  # you can change this to any integer
generator = torch.Generator("cuda").manual_seed(seed)

# Define prompts
positive_prompt = "Boy looking at computer screen, digital art, high detail, vibrant colors"
negative_prompt = "blurry, low quality, deformed, distorted, extra limbs, text, watermark"

# Generate image
image = pipeline(
    prompt=positive_prompt,
    negative_prompt=negative_prompt,
    generator=generator
).images[0]

# Show result
plt.imshow(image)
plt.axis("off")
plt.show()
