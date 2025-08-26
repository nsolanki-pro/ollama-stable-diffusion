import json
from diffusion_pipeline import generate_image
from ollama import AsyncClient
import asyncio
from PIL import Image
from io import BytesIO
import os
from datetime import datetime


# Helper function to load image bytes
def load_image_bytes(path):
    with Image.open(path) as img:
        byte_arr = BytesIO()
        img.save(byte_arr, format='PNG')
        return byte_arr.getvalue()


async def evaluate_images_text(positive_prompt: str, negative_prompt: str, image_path1: str, image_path2: str) -> dict:
    """
    Evaluate AI generated images via their description and the images themselves,
    and return parsed JSON containing positive and negative prompts.
    """
    client = AsyncClient()

    img_bytes1 = load_image_bytes(image_path1)
    img_bytes2 = load_image_bytes(image_path2)

    response = await client.chat(
        model="gemma3",
        messages=[{
            'role': 'user',
            'content': (
                f"The first image is original image."
                "Second image AI generated image with positive_prompt: '{positive_prompt}' & negative_prompt: '{negative_prompt}'. "
                "Describe in details & point by point what is good and bad in the second image compared to the first image."
                "Point out the differences between the two images in detail point by point."
                "Help me generate enhanced version of the original image by providing descriptive positive and negative prompts in JSON format like "
                '{"positive_prompt": [...], "negative_prompt": [...]}'
            ),
            'images': [img_bytes1, img_bytes2],
        }],
        options={
            'seed': 42,
            'temperature': 0.7,
        },
        keep_alive='1s'
    )

    # Extract JSON from model response
    content = response.message.content
    #print("Model response content:", content)  # Debug print
    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        json_text = content[start:end]
        prompts = json.loads(json_text)
    except (ValueError, json.JSONDecodeError) as e:
        print("Failed to parse JSON from model output:", e)
        prompts = {}

    return prompts


async def gen_image_prompt(image_path: str) -> dict:
    """
    Evaluate a generated image via its text prompt/description and the image itself (Base64),
    and return parsed JSON containing positive and negative prompts.
    """

    client = AsyncClient()

    img_bytes = load_image_bytes(image_path)


    response = await client.chat(
        model="gemma3",
        messages=[{
            'role': 'user',
            'content': (
                f"Generate descriptive prompts for reiki healing: "
                "1) Identify reiki point from image text. "
                "2) Describe healer’s posture for that point. "
                "3) Describe hand position for that point. "
                "4) Keep background plain, soft lighting, calming mood. "
                "5) Ensure correct posture & hands per reiki point. "
                "6) Limit prompt to 38 tokens. "
                "Output JSON with {'positive_prompt': [...], 'negative_prompt': [...]}"
            ),
            'images': [img_bytes],
        }],
        options={
            'seed': 42,
            'temperature': 0.7,
        },
        keep_alive='1s'
    )

    # Extract JSON from model response
    content = response.message.content
    #print("Model response content:", content)  # Debug print
    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        json_text = content[start:end]
        prompts = json.loads(json_text)
    except (ValueError, json.JSONDecodeError) as e:
        print("Failed to parse JSON from model output:", e)
        prompts = {}

    return prompts


async def refine_prompts(positive_prompt: str, negative_prompt: str, prompts_json: dict) -> dict:
    """
    Use the LLM to refine and improve the positive and negative prompts.
    """
    client = AsyncClient()

    # Combine original prompt and LLM-suggested prompts into instructions
    message_content = f"""
Original positive prompt: {positive_prompt}
Original negative prompt: {negative_prompt}

LLM suggested prompts: {json.dumps(prompts_json, indent=2)}

Please generate an improved version of the positive and negative prompts, optimized for image generation upto 38 tokens each, 
in JSON format like {{ "positive_prompts": [...], "negative_prompts": [...] }}.
    """

    response = await client.chat(
        model="gemma3",
        messages=[{'role': 'user', 'content': message_content}],
        options={
        'seed': 42,  # Set a specific seed for reproducible results
        'temperature': 0.7,
        },
        keep_alive='1s'  # Keep the connection alive for 1 second
    )

    # Parse JSON from response
    content = response.message.content
    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        json_text = content[start:end]
        improved_prompts = json.loads(json_text)
    except (ValueError, json.JSONDecodeError) as e:
        print("Failed to parse JSON from improved prompts:", e)
        improved_prompts = {}

    return improved_prompts

async def token_limit(prompt: str) -> dict:
    """
    Use the LLM to refine and improve the positive and negative prompts.
    """
    client = AsyncClient()

    # Combine original prompt and LLM-suggested prompts into instructions
    message_content = f"""
AI prompt: {prompt}
Make sure the prompt is within 38 tokens.
VERY IMPORTANT: Do not change the meaning of the prompt.
VERY IMPORTANT: Max token should be less than 38.
VERY IMPORTANT: Do not change the meaning of the prompt.
VERY IMPORTANT: Max token should be less than 38.
VERY IMPORTANT: Do not change the meaning of the prompt.
Please generate in JSON format like {{ "prompts": [...]}}.
    """

    response = await client.chat(
        model="gemma3",
        messages=[{'role': 'user', 'content': message_content}],
        options={
        'seed': 42,  # Set a specific seed for reproducible results
        'temperature': 0.7,
        },
        keep_alive='1s'  # Keep the connection alive for 1 second
    )

    # Parse JSON from response
    content = response.message.content
    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        json_text = content[start:end]
        improved_prompts = json.loads(json_text)
    except (ValueError, json.JSONDecodeError) as e:
        print("Failed to parse JSON from improved prompts:", e)
        improved_prompts = {}

    new_prompt = ", ".join(improved_prompts.get("prompts", []))
    print(f"Token limited prompt: {new_prompt} length: {len(new_prompt.split())} tokens")
    print(f"Prompt: {prompt} length: {len(prompt.split())} tokens")
    return new_prompt


async def main():
    # Iterate all images in inputs directory
    directory = "inputs"
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory, filename)
            print(f"Processing image: {image_path}")

            # Step 1: Generate initial prompts from the image
            initial_prompts = await gen_image_prompt(image_path)
            positive_prompt = initial_prompts.get("positive_prompt", "")
            negative_prompt = initial_prompts.get("negative_prompt", "")

            for i in range(2):
                print(f"\n--- Iteration {i+1} ---")

                print("Initial Prompts:", initial_prompts)
                print(f"Positive Prompt: {positive_prompt}")
                print(f"Negative Prompt: {negative_prompt}")
                #break
                # Step 2: Generate an image using the initial prompts
                generated_image = generate_image(positive_prompt, negative_prompt, seed=1234, save=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                generated_image_path = f"workdir/generated_{filename.split('.')[0]}_{i}_{timestamp}.png"
                generated_image.save(generated_image_path)
                print(f"Generated image saved to: {generated_image_path}")
                break
                # Step 3: Evaluate and refine prompts using the original and generated images
                evaluated_prompts = await evaluate_images_text(positive_prompt, negative_prompt, image_path, generated_image_path)
                print("Evaluated Prompts:", evaluated_prompts)

                refined_prompt = await refine_prompts(positive_prompt, negative_prompt, evaluated_prompts)
                print("Refined Prompts:", refined_prompt)

                # Step 4: Generate a new refined prompt.
                if refined_prompt:
                    # Join lists into comma-separated strings if needed
                    positive_prompt = ", ".join(refined_prompt.get("positive_prompts", []))
                    negative_prompt = ", ".join(refined_prompt.get("negative_prompts", []))
                else:
                    print("No refined prompts received, stopping iteration.")
                    break
                # Ensure prompts are within token limits
                positive_prompt = await token_limit(positive_prompt)
                negative_prompt = await token_limit(negative_prompt)
                #break
        break

if __name__ == "__main__":
    asyncio.run(main())
