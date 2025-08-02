import os
import pandas as pd
import random

CSV_PATH = os.path.join(os.path.dirname(__file__), "softskill_dataset.csv")

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError("Missing 'softskill_dataset.csv' in softskill_module.")

df = pd.read_csv(CSV_PATH)

# Experience mapping
EXPERIENCE_MAP = {
    "intern": "basic",
    "associate": "medium",
    "software engineer": "hard"
}

def get_question_by_experience(experience_input: str) -> str:
    experience_input = experience_input.lower().strip()
    if experience_input not in EXPERIENCE_MAP:
        raise ValueError("Invalid experience level. Choose from: intern, associate, software engineer")

    level = EXPERIENCE_MAP[experience_input]
    matching = df[df["Level"].str.lower() == level]

    if matching.empty:
        raise ValueError(f"No questions found for level '{level}'")
    
    return random.choice(matching["Question"].values)

def api_get_question_by_experience(experience_input: str) -> dict:
    try:
        question = get_question_by_experience(experience_input)
        return {"question": question}
    except ValueError as e:
        return {"error": str(e)}
