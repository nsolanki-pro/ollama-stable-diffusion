from diffusers import DiffusionPipeline
import torch
from datetime import datetime

def generate_image(positive_prompt: str, negative_prompt, seed: int = 42, save: bool = False):

    torch.cuda.empty_cache()
    torch.cuda.synchronize()  # Wait for all GPU operations to finish
    
    pipeline = DiffusionPipeline.from_pretrained(
        "SG161222/RealVisXL_V5.0",
        torch_dtype=torch.float16,
        safety_checker=None
    )
    pipeline.to("cuda")
    generator = torch.Generator("cuda").manual_seed(seed)

    image = pipeline(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        generator=generator,
        #cfg_scale=15.0,          # Higher CFG scale makes the model follow the prompt more strictly
        num_inference_steps=40,  # More steps usually produce more detailed and accurate images
        #guidance_rescale=0.7,    # Optional: can help make prompt adherence stronger without over-saturation
    ).images[0]

    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"workdir/generated_{timestamp}.png"
        image.save(save_path)

    # Manually free GPU memory
    del pipeline
    torch.cuda.empty_cache()
    torch.cuda.synchronize()  # Wait for all GPU operations to finish

    return image
