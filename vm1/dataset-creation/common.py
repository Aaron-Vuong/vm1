import os
import re

def get_home():
    HOME_PATH = os.getenv("HOME", None)
    assert HOME_PATH != None, "Need to know where models are stored!"
    return HOME_PATH

def create_custom_id(path: str, index: int): 
    custom_id = f"{os.path.basename(path)}_{index}"
    custom_id = custom_id.replace(".", "-").replace(" ", "_").replace("(", "").replace(")", "")
    return custom_id

def extract_option_answer(output_str):
    # Try to find the number within <answer> tags, if can not find, return None
    answer_pattern = r'<answer>\s*(\w+)\s*</answer>'
    match = re.search(answer_pattern, output_str)
    
    if match:
        return match.group(1)
    return None
