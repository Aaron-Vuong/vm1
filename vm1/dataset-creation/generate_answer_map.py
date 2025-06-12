import glob
import json
import os

import common

HOME_PATH = common.get_home()

modalities = glob.glob(f"{HOME_PATH}/Med-R1/Splits/modality/test/*.json")
question_types = glob.glob(f"{HOME_PATH}/Med-R1/Splits/question_type/test/*.json")

input_files = [*modalities, *question_types]

answers = {}

for json_f in input_files:
    with open(json_f, "r") as f:
        data = json.load(f)
    for index, qa in enumerate(data):
        custom_id = common.create_custom_id(json_f, index)
        answers[custom_id] = common.extract_option_answer(qa["solution"])

with open("answer_map.json", "w+", encoding="utf-8") as f:
    json.dump(answers, f, indent=2)
