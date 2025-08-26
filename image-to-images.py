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
    Optimized version:
    - Call 1: Describe detailed differences between original & enhanced image
    - Call 2: Generate enhanced positive prompts (photorealistic, aligned with original)
    - Call 3: Generate negative prompts (based on differences)
    """
    client = AsyncClient()

    img_bytes1 = load_image_bytes(image_path1)
    img_bytes2 = load_image_bytes(image_path2)

    # -------------------------------
    # Step 1: Get differences
    # -------------------------------
    diff_response = await client.chat(
        model="gemma3",
        messages=[{
            "role": "user",
            "content": (
                "Compare these two images.\n"
                "1) List point-by-point differences (style, posture, hands, background, artifacts).\n"
                "2) Return ONLY structured JSON like:\n"
                '{"differences": ["...","..."]}'
            ),
            "images": [img_bytes1, img_bytes2],
        }],
        options={"temperature": 0.3}
    )

    try:
        diff_text = diff_response.message.content
        start, end = diff_text.find("{"), diff_text.rfind("}") + 1
        differences = json.loads(diff_text[start:end])["differences"]
    except Exception as e:
        print("Error parsing differences:", e)
        differences = []

    # -------------------------------
    # Step 2: Generate Positive Prompt
    # -------------------------------
    pos_response = await client.chat(
        model="gemma3",
        messages=[{
            "role": "user",
            "content": (
                "Analyze the original image. Generate a short descriptive positive prompt "
                "to enhance realism and detail (max 38 tokens). "
                "Return JSON: {\"positive_prompt\": [\"...\"]}"
            ),
            "images": [img_bytes1],
        }],
        options={"temperature": 0.7}
    )

    try:
        pos_text = pos_response.message.content
        start, end = pos_text.find("{"), pos_text.rfind("}") + 1
        positive_prompts = json.loads(pos_text[start:end])["positive_prompt"]
    except Exception as e:
        print("Error parsing positive prompts:", e)
        positive_prompts = []

    # -------------------------------
    # Step 3: Generate Negative Prompt
    # -------------------------------
    neg_response = await client.chat(
        model="gemma3",
        messages=[{
            "role": "user",
            "content": (
                f"Based on these differences: {differences}\n"
                "Generate concise negative prompts to avoid these artifacts. "
                "Return JSON: {\"negative_prompt\": [\"...\"]}"
            )
        }],
        options={"temperature": 0.7}
    )

    try:
        neg_text = neg_response.message.content
        start, end = neg_text.find("{"), neg_text.rfind("}") + 1
        negative_prompts = json.loads(neg_text[start:end])["negative_prompt"]
    except Exception as e:
        print("Error parsing negative prompts:", e)
        negative_prompts = []

    return {
        "positive_prompt": positive_prompts,
        "negative_prompt": negative_prompts,
        "differences": differences
    }


async def refine_prompts(image_prompt: str, negative_prompt: str, prompts_json: dict, max_retries: int = 1) -> dict:
    """
    Refine and improve positive/negative prompts using LLM.
    Ensures photorealism, posture correctness (hands/legs), and <=38 tokens per prompt.
    Retries with stricter instructions if JSON parsing fails.
    """
    client = AsyncClient()

    def build_message(strict: bool = False) -> str:
        if not strict:
            return (
                f"Refine the following prompts for AI image generation:\n\n"
                f"Original positive prompt: {image_prompt}\n"
                f"Original negative prompt: {negative_prompt}\n\n"
                f"LLM suggested prompts:\n{json.dumps(prompts_json, indent=2)}\n\n"
                f"Requirements:\n"
                f"- Ensure correct and natural hand/leg posture\n"
                f"- Optimize wording for photorealistic results\n"
                f"- Each refined prompt <= 38 tokens\n"
                f"- Output ONLY JSON in format:\n"
                f'{{"positive_prompts": ["..."], "negative_prompts": ["..."]}}'
            )
        else:
            return (
                f"STRICT MODE: Output only valid JSON.\n"
                f"Refine the prompts into concise, photorealistic instructions (<=38 tokens each).\n"
                f"Ensure correct hand/leg posture.\n"
                f"Format must be exactly:\n"
                f'{{"positive_prompts": ["..."], "negative_prompts": ["..."]}}'
            )

    # Attempt loop
    for attempt in range(max_retries + 1):
        response = await client.chat(
            model="gemma3",
            messages=[{'role': 'user', 'content': build_message(strict=(attempt > 0))}],
            options={
                'seed': 42,
                'temperature': 0.7,
                'num_gpu': 99
            },
            keep_alive='1s'
        )

        content = response.message.content
        print(f"Refinement model response (attempt {attempt+1}):", content)

        try:
            start, end = content.find("{"), content.rfind("}") + 1
            json_text = content[start:end]
            improved_prompts = json.loads(json_text)

            # Clean up prompts
            improved_prompts["positive_prompts"] = [
                p.strip() for p in improved_prompts.get("positive_prompts", []) if p.strip()
            ]
            improved_prompts["negative_prompts"] = [
                p.strip() for p in improved_prompts.get("negative_prompts", []) if p.strip()
            ]

            return improved_prompts  # Success, return immediately

        except Exception as e:
            print(f"JSON parse failed on attempt {attempt+1}: {e}")
            if attempt == max_retries:
                # Final fallback
                return {"positive_prompts": [], "negative_prompts": []}


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
                f"Generate descriptive prompts: "
                "1) Identify scenario in the image. "
                "2) Describe in details unique aspect of the image. "
                "3) If applicable describe posture including hands legs."
                "4) Describe in details the background. "
                "5) Ensure correct posture & hands. "
                "6) Limit prompt to 38 tokens. "
                "Output JSON with {'positive_prompt': [...], 'negative_prompt': [...]}"
            ),
            'images': [img_bytes],
        }],
        options={
            'seed': 42,
            'temperature': 0.7,
            'num_gpu': 99
        },
        keep_alive='1s'
    )

    # Extract JSON from model response
    content = response.message.content
    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        json_text = content[start:end]
        prompts = json.loads(json_text)
    except (ValueError, json.JSONDecodeError) as e:
        print("Failed to parse JSON from model output:", e)
        prompts = {}

    return prompts


async def main():

    image_prompt = "1boy"
    negative_prompt = "bad quality, worst quality, low quality, lowres, normal quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands and fingers, poorly drawn hands and fingers, poorly drawn face, deformed, blurry, dehydrated, bad proportions, cloned face, disfigured, gross proportions, malformed limbs, missing arms and legs, fused fingers, too many fingers, long neck, photoshop"

    generated_image = generate_image(image_prompt, negative_prompt, seed=42, save=True)
    #generated_image.save("workdir/generated_initial.png")

    directory = "inputs"
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing image: {filename}")
            image_path = os.path.join(directory, filename)

            initial_prompts = await gen_image_prompt(image_path)
            positive_prompt = initial_prompts.get("positive_prompt", "")
            negative_prompt = initial_prompts.get("negative_prompt", "")
            print("Initial Positive Prompt:", positive_prompt)
            print("Initial Negative Prompt:", negative_prompt)

            for i in range(20):
                print(f"\n--- Iteration {i+1} ---")

                generated_image = generate_image(positive_prompt, negative_prompt, seed=42, save=False)
                timestamp = "{:02d}_{}".format(i, datetime.now().strftime("%Y%m%d_%H%M%S"))
                generated_image_path = f"workdir/generated_{filename.split('.')[0]}_{timestamp}.png"
                generated_image.save(generated_image_path)
                print(f"Generated image saved to: {generated_image_path}")


                # Step 3: Evaluate and refine prompts using the original and generated images
                evaluated_prompts = await evaluate_images_text(positive_prompt, negative_prompt, image_path, generated_image_path)
                print("Evaluated Prompts:", evaluated_prompts)

                refined_prompt = await refine_prompts(positive_prompt, negative_prompt, evaluated_prompts)
                print("Refined Prompts:", refined_prompt)

                # Step 4: Generate a new refined prompt.
                if refined_prompt:
                    positive_prompts = refined_prompt.get("positive_prompts", None)
                    negative_prompts = refined_prompt.get("negative_prompts", None)

                    positive_prompt = ", ".join(positive_prompts) if isinstance(positive_prompts, list) and len(positive_prompts) > 1 else (str(positive_prompts) if positive_prompts else "")
                    negative_prompt = ", ".join(negative_prompts) if isinstance(negative_prompts, list) and len(negative_prompts) > 1 else (str(negative_prompts) if negative_prompts else "")

                    print("Updated Positive Prompt:", positive_prompt)
                    print("Updated Negative Prompt:", negative_prompt)
                else:
                    print("No refined prompts received, stopping iteration.")
                    break
            break


if __name__ == "__main__":
    asyncio.run(main())
