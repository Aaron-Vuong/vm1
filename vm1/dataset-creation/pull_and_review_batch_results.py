import anthropic
import json
import logging
import sys

import common

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)
client = anthropic.Anthropic()

q_types = {
    'CTComputed_Tomography': {
        "path": 'modality/test/CT(Computed Tomography)_test.json',
        "type": "modality",
        "name": "CT"
    },
    'OCT_Optical_Coherence_Tomography': {
        "path": 'modality/test/OCT (Optical Coherence Tomography_test.json',
        "type": "modality",
        "name": "OCT",
    },
    'X-Ray': {
        "path": 'modality/test/X-Ray_test.json',
        "type": "modality",
        "name": "X-Ray"
    },
    'Dermoscopy': {
        "path": 'modality/test/Dermoscopy_test.json',
        "type": "modality",
        "name": "Dermoscopy"
    },
    'Microscopy_Images': {
        "path": 'modality/test/Microscopy Images_test.json',
        "type": "modality",
        "name": "Microscopy"
    },
    'Fundus_Photography': {
        "path": 'modality/test/Fundus Photography_test.json',
        "type": "modality",
        "name": "Fundus Photography"
    },
    'MR_Mag-netic_Resonance_Imaging': {
        "path": 'modality/test/MR (Mag-netic Resonance Imaging)_test.json',
        "type": "modality",
        "name": "MRI"
    },
    'ultrasound': {
        "path": 'modality/test/ultrasound_test.json',
        "type": "modality",
        "name": "Ultrasound",
    },
    'Modality_Recognition': {
        "path": 'question_type/test/Modality Recognition_test.json',
        "type": "task",
        "name": "Modality Recognition"
    },
    'Other_Biological_Attributes': {
        "path": 'question_type/test/Other Biological Attributes_test.json',
        "type": "task",
        "name": "Other Biological Attributes"
    },
    'Lesion_Grading': {
        "path": 'question_type/test/Lesion Grading_test.json',
        "type": "task",
        "name": "Lesion Grading"
    },
    'Anatomy_Identification': {
        "path": 'question_type/test/Anatomy Identification_test.json',
        "type": "task",
        "name": "Anatomy Identification"
    },
    'Disease_Diagnosis': {
        "path": 'question_type/test/Disease Diagnosis_test.json',
        "type": "task",
        "name": "Disease Diagnosis"
    }
}

PROMPT_JSON_PATH="/home/avuong/Med-R1/Splits/"
QUESTION_TEMPLATE = "{Question} Think through the question step by step in <think>...</think> tags. Then provide the correct single-letter choice (A, B, C, D,...) inside <answer>...</answer> tags."

def decode_custom_id(custom_id: str) -> tuple:
    index = custom_id.split("_")[-1]
    question_type = custom_id.split("_test-json")[0]
    info = q_types[question_type]
    return info, int(index)

with open("generated_batches.json", "r", encoding="utf-8") as f:
    batches = json.load(f)

ANSWER_TEMPLATE="""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{Question}<|im_end|>
<|im_start|>assistant
<|im_start|>{Answer}<|im_end|>"""

def assess_batch_accuracy(batches):
    global_correct = 0
    global_incorrect = 0
    correct_answers = []
    # Load the answer key.
    with open("answer_map.json", "r", encoding="utf-8") as f:
        answer_map = json.load(f)
    for batch in batches:
        correct = 0
        incorrect = 0
        for result in client.messages.batches.results(batch):
            q_type = result.custom_id.split("-json")[0]
            if result.result.type == "succeeded":
                model_answer = result.result.message.content[0].text
                # If the ground_truth == the model's answer, the model was correct.
                # Otherwise, it is incorrect and can be labelled more difficult.
                if answer_map[result.custom_id] == common.extract_option_answer(model_answer):
                    correct += 1
                    # Load the associated JSON and index.
                    info, question_index = decode_custom_id(result.custom_id)
                    with open(f"{PROMPT_JSON_PATH}/{info['path']}", "r", encoding="utf-8") as f:
                        data = json.load(f)
                        associated_q = data[question_index]

                    correct_answers.append({
                        "answer_idx": global_correct,
                        "id": result.custom_id,
                        "type": info["type"],
                        "category": info["name"],
                        "image": associated_q["image"],
                        "problem": associated_q["problem"],
                        "solution": associated_q["solution"],
                        "model_answer": model_answer,
                        "full_response": ANSWER_TEMPLATE.format(
                            Question=QUESTION_TEMPLATE.format(Question=associated_q["problem"]),
                            Answer=model_answer)
                    })
                    global_correct += 1
                else:
                    incorrect += 1
                    global_incorrect += 1
        logger.info("%s acc: %s/%s/%s", q_type, correct, incorrect, correct/(incorrect+correct))
    logger.info("Global Accuracy: %s/%s", global_correct, global_incorrect+global_correct)
    
    verified_reasoning_traces = correct_answers
    with open("verified_reasoning_traces.json", "w+", encoding="utf-8") as f:
        json.dump(verified_reasoning_traces, f, indent=2)

if __name__ == "__main__":
    assess_batch_accuracy(batches)
