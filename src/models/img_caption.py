import os
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
from config.prompts import img_caption_prompt

# Initialize the model and processor once to reuse across function calls.
MODEL_NAME = "unum-cloud/uform-gen2-qwen-500m"

model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)


def img_caption(img_path: str, max_new_tokens: int, temperature: float) -> str:
    """
    Generate an image caption for the image at the provided path.

    Parameters:
        img_path (str): The file path of the image.
        max_new_tokens (int): The maximum number of new tokens to generate.
        temperature (float): The sampling temperature.

    Returns:
        str: The generated caption.

    Raises:
        FileNotFoundError: If the image file at img_path does not exist.
        Exception: For any errors during image processing or model inference.
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"The image file {img_path} does not exist.")

    try:
        # Open and process the image using a context manager.
        with Image.open(img_path) as image:
            inputs = processor(
                text=[img_caption_prompt], images=[image], return_tensors="pt"
            )

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                do_sample=False,
                use_cache=True,
                max_new_tokens=max_new_tokens,
                eos_token_id=151645,
                pad_token_id=processor.tokenizer.pad_token_id,
                temperature=temperature,
            )

        prompt_len = inputs["input_ids"].shape[1]
        decoded_text = processor.batch_decode(output[:, prompt_len:])[0]
        return decoded_text

    except Exception as e:
        raise Exception(f"Error during caption generation: {e}")
