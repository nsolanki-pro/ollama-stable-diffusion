import json
from diffusion_pipeline import generate_image
from ollama import AsyncClient
import asyncio
from PIL import Image
from io import BytesIO
from datetime import datetime

async def evaluate_image_text(prompt_description: str, image: Image.Image) -> dict:
    """
    Evaluate a generated image via its text prompt/description and the image itself (Base64),
    and return parsed JSON containing positive and negative prompts.
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
            'content': """This is AI generated image. Help me with generating a positive and a negative prompts for it in json format  like {{ "positive_prompts": [...], "negative_prompts": [...] }}.""",
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


async def refine_prompts(image_prompt: str, negative_prompt: str, prompts_json: dict) -> dict:
    """
    Use the LLM to refine and improve the positive and negative prompts.
    """
    client = AsyncClient()

    # Combine original prompt and LLM-suggested prompts into instructions
    message_content = f"""
Original positive prompt: {image_prompt}
Original negative prompt: {negative_prompt}

LLM suggested prompts: {json.dumps(prompts_json, indent=2)}

Please generate an improved version of the positive and negative prompts, optimized for image generation upto 77 tokens each, 
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


async def main():
    # Generate image as PIL.Image
    #image_prompt = "Aerial view of a pristine archipelago, turquoise waters dotted with lush emerald islands. Intricate coral reefs visible beneath crystal-clear shallows, forming mesmerizing patterns. White sandy beaches curve along island edges, contrasting with dense tropical forests. Dramatic cliffs on larger islands, their faces etched with cascading waterfalls. Sunlight glimmers off wave crests, creating a sparkling tapestry across the ocean surface. Clouds cast soft shadows, adding depth to the scene. Rich biodiversity evident in the vibrant colors of flora and fauna. 8K resolution, hyper-detailed, masterpiece quality, perfect lighting, intricate textures."
    #image_prompt = "macro photo, ant lifting a tiny dumbbell, bodybuilding in a miniature gym, high detail, humorous and cute"
    image_prompt = "Mid-shot photograph, woman with short dark hair wearing a cream-colored turtleneck sweater and dark wash blue jeans, seated on a simple wooden chair with a slightly worn finish, bare feet on a textured beige rug, plain yellow wall in the background, soft, diffused lighting, shallow depth of field, minimalist aesthetic, 8k resolution, photorealistic"
    #negative_prompt = "blurry, low quality, deformed, distorted, extra limbs, text, watermark"
    negative_prompt = "cartoon, illustration, painting, sketch, anime, overly stylized, vibrant colors, bright lighting, sharp focus throughout, dramatic lighting, cluttered background, outdoor setting, action shot, overly complex composition, fantasy elements, blurry"

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
        prompts_json = await evaluate_image_text(image_prompt, image)
        #print("Extracted prompts JSON:", prompts_json)

        # Refine prompts for better image generation
        improved_prompts = await refine_prompts(image_prompt, negative_prompt, prompts_json)
        #print("Improved prompts JSON:", improved_prompts)
        # Generate a new image using improved prompts
        if improved_prompts:
            # Join lists into comma-separated strings if needed
            new_positive_prompt = ", ".join(improved_prompts.get("positive_prompts", []))
            new_negative_prompt = ", ".join(improved_prompts.get("negative_prompts", []))

        # Print all prompts for reference
        print("Current Positive Prompt:", new_positive_prompt)
        print("Current Negative Prompt:", new_negative_prompt)
        print(" original Positive Prompt:", image_prompt)
        print(" original Negative Prompt:", negative_prompt)
        #break


if __name__ == "__main__":
    asyncio.run(main())
