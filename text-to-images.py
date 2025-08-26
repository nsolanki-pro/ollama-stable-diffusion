import json
from diffusion_pipeline import generate_image
from ollama import AsyncClient
import asyncio
from PIL import Image
from io import BytesIO
from datetime import datetime

async def evaluate_image_text(positive_prompt: str, negative_prompt: str, image: Image.Image) -> dict:
    """
    Evaluate a generated image via its text prompt/description and the image itself (Base64),
    and return parsed JSON containing positive and negative prompts in JSON format like {{ "positive_prompt": [...], "negative_prompt": [...] }}.
    """
    client = AsyncClient()

    # Convert PIL image to bytes
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()

    response = await client.chat(
        model="gemma3",
        messages=[{
            'role': 'user',
            'content': """
            Original positive prompt: {positive_prompt}
            Original negative prompt: {negative_prompt}
            Check the image attached & see if the image matches the prompt.
            Generate new enhanced prompts for image generation that would align with the original prompt but better image output.
            Help me with generating a positive and a negative prompts aligned with original prompt in json format  like {{ "positive_prompt": [...], "negative_prompt": [...] }}.
            Note: Make sure each prompt is upto 70 word only.""",
            'images': [img_bytes],
        }],
        options={
        'seed': 42,  # Set a specific seed for reproducible results
        'temperature': 0.7,
        },
        keep_alive='1s'  # Keep the connection alive for 1 second
    )

    # Attempt to extract JSON from the model's response
    content = response.message.content
    try:
        # Find the first { ... } in the text
        start = content.find("{")
        end = content.rfind("}") + 1
        json_text = content[start:end]
        prompts = json.loads(json_text)
    except (ValueError, json.JSONDecodeError) as e:
        print("Failed to parse JSON from model output:", e)
        prompts = {}

    #print("Evaluated Prompts:", prompts)
    return prompts

async def main():
    image_prompt = "A young boy with messy brown hair, standing with his arms crossed, wearing a worn leather jacket, standing in a magical forest, glowing mushrooms, whimsical art style."
    negative_prompt = "blurry, low quality, deformed, distorted, extra limbs, text, watermark cartoon, illustration, painting, sketch, anime, overly stylized, vibrant colors, bright lighting, sharp focus throughout, dramatic lighting, cluttered background, outdoor setting, action shot, overly complex composition, fantasy elements, blurry"

    new_positive_prompt = image_prompt
    new_negative_prompt = negative_prompt

    for i in range(10):
        print(f"--- Iteration {i+1} ---")

        image = generate_image(new_positive_prompt, new_negative_prompt, seed=42)

        timestamp = "{:02d}_{}".format(i, datetime.now().strftime("%Y%m%d_%H%M%S"))
        generated_image_path = f"workdir/generated_{timestamp}.png"
        image.save(generated_image_path)
        print(f"Generated image saved to: {generated_image_path}")

        # Evaluate using both prompt and image
        improved_prompts = await evaluate_image_text(image_prompt, negative_prompt, image)
        #print("Extracted prompts JSON:", prompts_json)

        if improved_prompts:
            # Join lists into comma-separated strings if needed
            new_positive_prompt = improved_prompts.get("positive_prompt", [])
            new_negative_prompt = improved_prompts.get("negative_prompt", [])
            new_positive_prompt = ", ".join(new_positive_prompt) if isinstance(new_positive_prompt, list) and len(new_positive_prompt) > 1 else (str(new_positive_prompt) if new_positive_prompt else "")
            new_negative_prompt = ", ".join(new_negative_prompt) if isinstance(new_negative_prompt, list) and len(new_negative_prompt) > 1 else (str(new_negative_prompt) if new_negative_prompt else "")

        # Print all prompts for reference
        print("Current Positive Prompt:", new_positive_prompt)
        print("Current Negative Prompt:", new_negative_prompt)
        print(" original Positive Prompt:", image_prompt)
        print(" original Negative Prompt:", negative_prompt)


if __name__ == "__main__":
    asyncio.run(main())
