import nltk
from nltk.tokenize import sent_tokenize
import language_tool_python
from sentence_transformers import SentenceTransformer, util
import subprocess
import os
import re

nltk.download('punkt')

# NLP Tools
tool = language_tool_python.LanguageTool('en-US')
model = SentenceTransformer('all-MiniLM-L6-v2')

def grammar_score(text):
    matches = tool.check(text)
    error_count = len(matches)
    score = 1 / (1 + error_count)
    return round(score, 4), error_count

def clarity_score(text):
    sentences = sent_tokenize(text)
    sentence_lengths = [len(s.split()) for s in sentences]
    if not sentence_lengths:
        return 0.0
    avg_length = sum(sentence_lengths) / len(sentence_lengths)
    if avg_length < 5:
        return 0.4
    elif avg_length > 25:
        return 0.6
    else:
        return 1.0

def coherence_score(text):
    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        return 1.0
    embeddings = model.encode(sentences, convert_to_tensor=True)
    sim_sum = 0.0
    for i in range(len(sentences) - 1):
        sim = util.cos_sim(embeddings[i], embeddings[i + 1]).item()
        sim_sum += sim
    return round(sim_sum / (len(sentences) - 1), 4)

def evaluate_communication(text):
    g_score, g_errors = grammar_score(text)
    c_score = clarity_score(text)
    coh_score = coherence_score(text)
    final_score = round((g_score + c_score + coh_score) / 3, 4)
    return {
        "Grammar Score": g_score,
        "Grammar Errors": g_errors,
        "Clarity Score": c_score,
        "Coherence Score": coh_score,
        "Final Communication Score": final_score
    }

def get_label(score):
    if score >= 0.85:
        return "Excellent Communicator"
    elif score >= 0.7:
        return "Good Communicator"
    elif score >= 0.5:
        return "Average Communicator"
    else:
        return "Needs Improvement"

def evaluate_answer_relevance(question, answer):
    try:
        result = subprocess.run(
            ["node", os.path.join(os.path.dirname(__file__), "relevance_gpt.js"), question, answer],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"❌ GPT Evaluation Error:\n{e.stderr.strip()}"

def adjust_final_score_based_on_relevance(final_score: float, relevance_feedback: str) -> float:
    match = re.search(r"Relevance Score: (\d)/5", relevance_feedback)
    if not match:
        # Relevance score missing or GPT failed → penalize
        return 0.2

    relevance = int(match.group(1)) / 5.0
    if relevance < 0.4:
        return 0.2  # Irrelevant
    return final_score  # Relevant, keep original score

