"""
Script to create a batch file, randomized.

To replicate the original batch file, use the seed 42.
"""
import base64
import glob
import json
import jsonlines
import logging
import os
import PIL
import random
import sys

from openai import OpenAI
from PIL import Image

import common

HOME_PATH = common.get_home()

#client = OpenAI()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Number of training samples to take from each modality + question type.
NUM_SAMPLES_PER_JSON = 100000

# Model to generate with.
MODEL="gpt-4o"
# Maximum limit of tokens.
TOKEN_LIMIT=2048

random.seed(42)

BASE_PATH=f"{HOME_PATH}/Med-R1/OmniMedVQA"
OUTPUT_DIR="medr1_json"
OUTPUT_JSONL_DIR="medr1_jsonl"
CONVERTED_IMAGES="converted_images"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_JSONL_DIR, exist_ok=True)
os.makedirs(CONVERTED_IMAGES, exist_ok=True)

modalities = glob.glob(f"{HOME_PATH}/Med-R1/Splits/modality/test/*.json")
question_types = glob.glob(f"{HOME_PATH}/Med-R1/Splits/question_type/test/*.json")

QUESTION_TEMPLATE = "{Question} Think through the question step by step in <think>...</think> tags. Provide reasoning as to why each option is incorrect or incorrect as part of the thinking process. Revisit your reasoning 1 to 5 times, starting each revisit with 'Wait'. Reason until you reach at least 2000 tokens. Then provide the correct single-letter choice (A, B, C, D,...) inside <answer>...</answer> tags."

def convert_to_jpg(image_path, output_path):
    try:
        img = Image.open(image_path)
        img = img.convert('RGB') # Ensure image is in RGB mode
        # Minimally downsize the image to fit image requirements of different APIs.
        downscaled_size = (int(img.size[0] * 0.8), int(img.size[1] * 0.8))
        logger.info(f"Downscaled Size: {downscaled_size}")
        img.resize(downscaled_size, PIL.Image.LANCZOS)
        img.save(output_path, 'JPEG', optimize=True, quality=85)
        logger.info(f"Converted {image_path} to {output_path}")
    except Exception as e:
        logger.error(f"Error converting {image_path}: {e}")
        raise e

def encode_image_to_base64(image_path):
  """Encodes an image to a Base64 string.

  Args:
    image_path: The path to the image file.

  Returns:
    A string containing the Base64 representation of the image, or None if an error occurs.
  """
  try:
    with open(image_path, "rb") as image_file:
      image_data = image_file.read()
      base64_encoded_string = base64.b64encode(image_data)
      base64_string = base64_encoded_string.decode('utf-8')
      return base64_string
  except FileNotFoundError:
    logger.error(f"Error: Image file not found at {image_path}")
    return None
  except Exception as e:
    logger.error(f"An error occurred: {e}")
    return None

def create_batch_req(request_id: str, data: dict, provider: str="anthropic") -> dict:
    """
    Creates a Batch API compatible JSON request object.

    Args:
        request_id: The custom ID to mark this request with.
        data: The data to embed into the batch request.
        provider: The provider we want to format the batch requests to.

    Returns:
        A JSON object containing the batch request.
    
    """ 
    if provider == "openai":
        return {
            "custom_id": request_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL,
                "max_tokens": TOKEN_LIMIT,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a skilled diagnostician."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "file",
                                "file": {
                                    "file_id": f"data:image/{data['image_type']};base64,{data['base64']}"
                                }
                            },
                            {
                                "type": "text",
                                "text": QUESTION_TEMPLATE.format(Question=data["problem"])
                            }
                        ]
                    }
                ]
            }
        }
    elif provider == "anthropic":
        if data['image_type'] == "jpg":
            data['image_type'] = "jpeg"
        return {
            "custom_id": request_id,
            "model": "claude-sonnet-4-20250514",
            "max_tokens": TOKEN_LIMIT,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": f"image/{data['image_type']}",
                                "data": data["base64"],
                            },
                        },
                        {
                            "type": "text",
                            "text": QUESTION_TEMPLATE.format(Question=data["problem"])
                        }
                    ]
                }
            ]
        }

def combine_batch_files(input_files: list):
    """
    Combine all sub batch files into a single file.

    Args:
        input_files: Batch files to concatenate.
    """
    logger.info(f"Files: {input_files}")
    all_data = []
    for json_f in input_files:
        with open(json_f, "r") as f:
            data = json.load(f)
        with jsonlines.open(f"{OUTPUT_JSONL_DIR}/{os.path.basename(json_f)}l", "w") as writer:
            writer.write_all(data)
        all_data.extend(data)
    
    # Dump the combined batch file to send to the OpenAI Batch API.
    with jsonlines.open("combined_batch_file.jsonl", "w") as writer:
        writer.write_all(all_data)

def main(input_files: list):
    batch_files = []
    for json_f in input_files:
        collected_train_split = []
        train_dataset = []
        test_dataset = []
        with open(json_f, "r") as f:
            data = json.load(f)
        
        # Create a set of available indices for this JSON.
        available_indices = set([i for i in range(0, len(data))])

        for i in range(min(NUM_SAMPLES_PER_JSON, len(data))):
            image_type = "jpg"
            index = random.choice(list(available_indices))
            available_indices.remove(index)
            
            # Add this VQA to a training dataset.
            train_dataset.append(data[index])
            # Upload any images that are needed in the train dataset.
            image_path = f"{BASE_PATH}/{data[index]['image']}"
            logger.info("Uploading %s (%d/%d)", os.path.basename(image_path), i, NUM_SAMPLES_PER_JSON)
            # Convert to JPG for non-PDF or non-JPG formats. Commonly, we can get things like BMP or TIF.
            if image_path.lower().endswith(('.png')):
                image_type="png"
            elif image_path.lower().endswith(('.jpeg', 'jpg')):
                image_type="jpg"
            else:
                image_type="jpg"
                new_image_path=f"{CONVERTED_IMAGES}/{os.path.basename(image_path) + '.jpg'}"
                convert_to_jpg(image_path, new_image_path)
                image_path = new_image_path
            base64_str = encode_image_to_base64(image_path)
            data[index]["image_type"] = image_type
            data[index]["base64"] = base64_str
            # Append the batch file JSON request.
            if data[index].get("id", None) is None:
                custom_id = common.create_custom_id(json_f, index)
            else:
                # Get the ID if the data has a unique ID already.
                custom_id = data[index]["id"]
            collected_train_split.append(create_batch_req(custom_id, data[index]))

        # Ensure that we also record a corresponding test split
        # that isn't included in the training dataset.
        for i in available_indices:
            test_dataset.append(data[i])
        
        assert len(data) == len(test_dataset) + len(train_dataset), f"Length of generated datasets does not match the original dataset? {len(data)} != {len(test_dataset) + len(train_dataset)}"

        new_f = os.path.basename(json_f).replace("_test.json", "")
        # Dump the test dataset, we'll need it for evaluation later.
        with open(f"{OUTPUT_DIR}/{new_f}_newtest.json", "w+", encoding="utf-8") as f:
            json.dump(test_dataset, f, indent=2)
            
        # Dump the training dataset for our records.
        with open(f"{OUTPUT_DIR}/{new_f}_newtrain.json", "w+", encoding="utf-8") as f:
            json.dump(train_dataset, f, indent=2)

        # Dump the sub batch file in-case we must send smaller batches to the OpenAI API.
        with open(f"{OUTPUT_DIR}/{new_f}_batch_file.json", "w+", encoding="utf-8") as f:
            json.dump(collected_train_split, f, indent=2)
        
        batch_files.append(f"{OUTPUT_DIR}/{new_f}_batch_file.json")

    combine_batch_files(batch_files)

if __name__ == "__main__":
    # Clear uploaded files.
    # DON'T UNCOMMENT, THIS DELETES ALL OF YOUR UPLOADED FILES TO OPENAI!
    # uploaded_files = client.files.list()
    # for f in uploaded_files:
    #   client.files.delete(f.id)
    # sys.exit(0)
    main(["medr1_results/train_split.json"])#[*modalities, *question_types])
