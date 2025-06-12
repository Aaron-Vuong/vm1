<div align="center">
<h1>
  <b>vm1</b>: Enhancing Reasoning Capabilities of Medical Vision-Language Models
</h1>
<p>
Extending the s1 test-time scaling methodology to medical vision large language models.
</p>
</div>

### Installation
The following snippet installs the necessary packages to run inferencing locally.
If you are only doing the dataset creation locally and inferencing on Colab, utilize the Colab
notebooks instead and only install `anthropic` + `openai`.

```bash
conda create -n vm1
conda activate vm1

bash setup.sh
```

### Replicating Results
To reproduce the dataset creation process, you can follow the code snippet below.
`$REPO_ROOT` refers to the directory this README.md resides in.
All scripts rely on the following repositories being in your `$HOME` directory.
- [Med-R1](https://huggingface.co/yuxianglai117/Med-R1)
- [OmniMedVQA](https://huggingface.co/datasets/foreverbeliever/OmniMedVQA/tree/main)
- [Med-R1 (Github)](https://github.com/Yuxiang-Lai117/Med-R1/)

Do some initial setup and process the `output.json` that you get from running Med-R1 in `replication/Med_R1.ipynb`.
```bash
# Start in the $REPO_ROOT.
cd $REPO_ROOT

# Copy all of the necessary repositories to $HOME.
# We especially need OmniMedVQA to encode the images to send to APIs.
# We also need Med-R1's Github to reference the test split
bash vm1/copy_repositories.sh

# Go to the dataset-creation directory and perform necessary steps.
cd vm1/dataset-creation

# Copy over the train_split.json that you generated in replication/
cd medr1_results
# COPY the output.json that you get from running the replication/ .ipynb script in Colab here!
# scp output.json ./

# Filter for only Med-R1 failures.
python3 extract_failed.py 
# Split the resulting dataset 50/50.
python3 generate_splits.py

# You should see a test_split.json and train_split.json with VQA pairs inside of them.
```

Then, we can generate batch files, send them to the API to generate reasoning traces, and produce a dataset.
```bash
# Start in the dataset-creation directory.
cd $REPO_ROOT/vm1/dataset-creation

# Create a batch file from the JSON file (train_split.json) that it points to.
# Default Batch API is Anthropic. OpenAI is untested.
python3 create_batch_file.py

# Splits the batch file into manageable chunks so we aren't rate-limited.
python3 split_batch_file.py

# Sends the requests to the Claude API
python3 batch_reasoning_trace_gen.py

# Produces a dataset similar to avuong/vm1
python3 pull_and_review_batch_results.py
```

Create a new huggingface dataset at [HF-Create a new dataset repository](https://huggingface.co/new-dataset).
Then, upload the `verified_reasoning_traces.json` to the dataset directory.

Once we have a dataset, we can SFT the model. Run the `training/SFT.ipynb` script in Colab using an A100 GPU runtime.
This will also auto-upload a new model called `vm1-sft`.

We can now refer to the new model in `inference/Inference.ipynb` and begin to run queries on it.

### Models and Data
| Model       | Base Model           | Training Data                                     | Link                                             |
| ----------- | -------------------- | ------------------------------------------------- | ------------------------------------------------ |
| **vm1-sft** | Med-R1 (Qwen2-VL-2B) | [vm1](https://huggingface.co/datasets/avuong/vm1) | [HF Link](https://huggingface.co/avuong/vm1-sft) |

### Inference

vm1 can be directly inferenced with the following snippet, similar to Med-R1's inferencing code.
```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer

MODEL_PATH="avuong/vm1-sft"
BASE_PATH="OmniMedVQA/"
QUESTION_TEMPLATE = "{Question} First output the thinking process in <think> </think> and final choice (A, B, C, D ...) in <answer> </answer> tags."

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(MODEL_PATH)

i = {
    "image": "Images/Chest CT Scan/test/adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib/000139 (9).png",
    "problem": "What imaging technique is employed for obtaining this image? A)Mammogram, B)Positron emission tomography (PET), C)CT, D)Fluoroscopy",
    "solution": "<answer> C </answer>"
}

message = {
    "role": "user",
    "content": [
        {
            "type": "image",
            "image": f"file://{BASE_PATH}{i['image']}"
        },
        {
            "type": "text",
            "text": QUESTION_TEMPLATE.format(Question=i['problem'])
        }
    ]
}

image_inputs, video_inputs = process_vision_info(message)
inputs = processor(
    text=text,
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(
    **inputs,
    use_cache=True,
    max_new_tokens=max_new_tokens,
    do_sample=False)

generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
batch_output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print(batch_output_text)
```
