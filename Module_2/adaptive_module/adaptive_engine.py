import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer, util

# Load model and data
MODEL_PATH = "Module_2/adaptive_module/next_complexity_model.pkl"
DATA_PATH = "Module_2/adaptive_module/model_b_full_labels.csv"

model = joblib.load(MODEL_PATH)
question_bank = pd.read_csv(DATA_PATH)
semantic_model = SentenceTransformer("all-mpnet-base-v2")

# Map experience to encoded value and starting score
experience_mapping = {
        "Intern": 0,
        "Associate": 1,
        "Engineer": 2
}

experience_starting_score = {
    "Intern": 2.0,
    "Associate": 2.7,
    "Engineer": 3.2,
}

def get_initial_complexity(experience):
    return experience_starting_score.get(experience, 2.0)

def encode_experience(experience):
    return experience_mapping.get(experience, 0)

def get_question_by_complexity(skills, target_complexity, asked_ids):
    subset = question_bank[
        question_bank["technology"].str.lower().isin([s.lower() for s in skills])
    ].copy()
    subset["score_diff"] = abs(subset["complexity_score"] - target_complexity)
    subset = subset[~subset.index.isin(asked_ids)]

    if subset.empty:
        return None

    row = subset.sort_values("score_diff").iloc[0]
    return {
        "id": str(row.name),
        "question": row["question_text"],
        "expected_answer": row["expected_answer"],
        "complexity": row["complexity_score"],
        "technology": row["technology"]
    }

def evaluate_answer(expected_answer, candidate_answer):
    emb_expected = semantic_model.encode(expected_answer, convert_to_tensor=True)
    emb_candidate = semantic_model.encode(candidate_answer, convert_to_tensor=True)
    similarity = util.cos_sim(emb_expected, emb_candidate).item()
    return round(similarity, 2)

def predict_next_complexity(question_text, expected_answer, correctness, current_complexity, experience_encoded):
    qa_text = question_text + " " + expected_answer
    sample = pd.DataFrame([{
        "qa_text": qa_text,
        "complexity_score": current_complexity,
        "answer_quality_score": correctness,
        "experience_encoded": experience_encoded
    }])
    return model.predict(sample)[0]
