import anthropic
import glob
import json
import logging
import os
import sys
import time
from openai import OpenAI
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)
#openai_client = OpenAI()

# Upload the batch input file to OpenAI Files API.

def run_completions():
    """
    Method to run via the /chat/completions API.
    """
    output_log = "responses_log.txt"
    output_json = "responses.json"
    json_files = glob.glob("json/*_batch_file.json")
    ignore_paths = [
        "sample_batch_file.json",
        "combined_batch_file.json"
    ]

    for file_path in json_files:
        if file_path in ignore_paths:
            continue
        logger.info("Sending requests from %s", file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for req in data:
            body = req["body"]
            response = openai_client.chat.completions.create(
                model=body["model"],
                messages=body["messages"],
                max_tokens=body["max_tokens"]
            )
            with open(output_log, "a+", encoding="utf-8") as f:
                f.write(f"{req.custom_id}: {response.id}\n")
            with open(f"{output_json}l", "a+", encoding="utf-8") as f:
                json.dump(response, f)
                f.write("\n")

            exit(0)

def run_batch(provider: str="anthropic"):
    """
    Method to run via the Batch API.
    """
    if provider == "openai":
        jsonl_files = glob.glob("jsonl/*")

        for f in jsonl_files:
            batch_input_file = client.files.create(
                file=open(f, "rb"),
                purpose="batch"
            )
            logger.info(f"Uploaded batch file: {batch_input_file}")

            # Create the batch job.
            batch_input_file_id = batch_input_file.id
            batch_req = openai_client.batches.create(
                input_file_id=batch_input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                    "description": "Batch job to create reasoning traces."
                }
            )
            logger.info("%s: %s", f, batch_input_file_id)

            # Look at https://platform.openai.com/batches/ to track them.
    elif provider == "anthropic":
        client = anthropic.Anthropic()
        json_files = glob.glob("medr1_json/*_batch_file*.json")
        ignore_paths = [
            "sample_batch_file.json",
            "combined_batch_file.json"
            "medr1_json/train_split.json_batch_file.json"
        ]
        batches = []
        for file_path in json_files:
            all_requests = []
            if file_path in ignore_paths:
                continue
            logger.info("Sending requests from %s", file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for req in data:
                all_requests.append(
                    Request(
                        custom_id=req["custom_id"],
                        params=MessageCreateParamsNonStreaming(
                            model=req["model"],
                            max_tokens=req["max_tokens"],
                            messages=req["messages"]
                        )
                    )
                )
            # Make a batch per file, it may be too large otherwise.
            msg_batch = client.messages.batches.create(requests=all_requests)
            batches.append(msg_batch.id)
            logger.info("%s: %s", file_path, msg_batch.id)
        with open("generated_batches.json", "w+", encoding="utf-8") as f:
            json.dump(batches, f, indent=2)
            
        logger.info(json.dumps(batches, indent=2))
    else:
        raise Exception("No matching provider?")

#run_completions()
run_batch()
