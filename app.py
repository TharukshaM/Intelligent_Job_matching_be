class ModelPerformance:
    def __init__(self, *args, **kwargs):
        pass

from flask import Flask, request, jsonify
from Module_2.softskill_module.evaluator import evaluate_communication, get_label, evaluate_answer_relevance, adjust_final_score_based_on_relevance
from Module_2.softskill_module.question_selector import api_get_question_by_experience
from Module_2.adaptive_module.adaptive_engine import (
    question_bank,
    get_initial_complexity, encode_experience,
    get_question_by_complexity, evaluate_answer, predict_next_complexity
)
from Module_2.adaptive_module.session_store import create_session, get_session, update_session
from flask import Flask
from flask_cors import CORS

# Module 1 dependancies
import joblib
from Module_1.feature_extractor.extract_features import CodeFeatureExtractor
import requests
import os
from dotenv import load_dotenv
from Module_1.github_utils.github_api import get_user_repos, get_java_js_files, fetch_code_from_url
from Module_1.algorithm_detection.detector import analyze_git_commit_chunk
from Module_1.extractors.similarity_checker import compute_code_similarity
from Module_1.extractors.graph_semantic_similarity import GraphSemanticSimilarity
from Coding_Question.code_question_gpt import generate_question_and_refs

load_dotenv()

app = Flask(__name__)
CORS(app)

@app.route("/api/get_question", methods=["POST"])
def get_question():
    data = request.get_json()
    experience = data.get("experience")

    if not experience:
        return jsonify({"error": "The 'experience' field is required."}), 400

    response = api_get_question_by_experience(experience)
    status_code = 200 if "question" in response else 400
    return jsonify(response), status_code

@app.route("/api/evaluate_softskill", methods=["POST"])
def evaluate_softskill():
    data = request.get_json()
    answer = data.get("answer")
    question = data.get("question")

    if not question or not answer:
        return jsonify({"error": "Both 'question' and 'answer' fields are required."}), 400

    try:
        scores = evaluate_communication(answer)
        relevance_feedback = evaluate_answer_relevance(question, answer)
        adjusted_score = adjust_final_score_based_on_relevance(scores["Final Communication Score"], relevance_feedback)
        scores["Final Communication Score"] = round(adjusted_score, 4)
        label = get_label(adjusted_score)

        return jsonify({
            "question": question,
            "answer": answer,
            "scores": scores,
            "label": label,
            "relevance_feedback": relevance_feedback
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/api/start_adaptive_assessment", methods=["POST"])
def start_adaptive():
    data = request.get_json()
    experience = data.get("experience")
    skills = data.get("skills")

    if not experience or not skills:
        return jsonify({"error": "Missing experience or skills"}), 400

    session_id = create_session(experience, skills)
    experience_encoded = encode_experience(experience)
    current_complexity = get_initial_complexity(experience)

    update_session(session_id, {
        "experience_encoded": experience_encoded,
        "current_complexity": current_complexity
    })

    session = get_session(session_id)
    question = get_question_by_complexity(skills, current_complexity, [])

    if not question:
        return jsonify({"error": "No matching questions found"}), 404

    session["asked_questions"].append(int(question["id"]))
    update_session(session_id, session)

    return jsonify({
        "session_id": session_id,
        "question": {
            "id": question["id"],
            "text": question["question"],
            "complexity": question["complexity"],
            "technology": question["technology"]
        }
    })

@app.route("/api/submit_adaptive_answer", methods=["POST"])
def submit_adaptive_answer():
    data = request.get_json()
    session_id = data.get("session_id")
    question_id = int(data.get("question_id"))
    candidate_answer = data.get("candidate_answer")

    session = get_session(session_id)
    if not session:
        return jsonify({"error": "Invalid session ID"}), 404

    # Get current question from question bank
    question_row = question_bank.loc[question_id]
    expected = question_row["expected_answer"]
    question_text = question_row["question_text"]
    complexity = question_row["complexity_score"]

    correctness = evaluate_answer(expected, candidate_answer)
    next_complexity = predict_next_complexity(
        question_text, expected, correctness,
        complexity, session["experience_encoded"]
    )

    # Update session
    session["current_complexity"] = next_complexity
    session["score_history"].append({
        "question_id": question_id,
        "correctness": correctness,
        "complexity": complexity
    })
    update_session(session_id, session)

    # Get next question
    next_q = get_question_by_complexity(
        session["skills"], next_complexity, session["asked_questions"]
    )
    if next_q:
        session["asked_questions"].append(int(next_q["id"]))
        update_session(session_id, session)
        return jsonify({
            "next_question": {
                "id": next_q["id"],
                "text": next_q["question"],
                "complexity": next_q["complexity"],
                "technology": next_q["technology"]
            }
        })
    else:
        return jsonify({"message": "No more questions available"}), 200


@app.route("/api/finalize_adaptive_assessment", methods=["POST"])
def finalize_adaptive():
    data = request.get_json()
    session_id = data.get("session_id")
    session = get_session(session_id)

    if not session:
        return jsonify({"error": "Invalid session ID"}), 404

    history = session["score_history"]
    avg_score = round(sum([s["correctness"] for s in history]) / len(history), 3) if history else 0.0

    return jsonify({
        "final_score": avg_score,
        "total_questions": len(history),
        "details": history
    })

# Module_1 Routes
# Load model once on startup
bundle = joblib.load("./Module_1/mccall_trained_models.joblib")
model = bundle['models']['overall_quality']
feature_names = [
    'cyclomatic_complexity',
    'num_functions',
    'num_classes',
    'comment_ratio',
    'max_nesting_depth',
    'has_error_handling',
    'uses_framework',
    'num_imports',
    'avg_function_length',
    'has_recursion',
    'max_line_length',
    'is_object_oriented',
    'uses_logging'
]

extractor = CodeFeatureExtractor()
code_assessment_analyzer = GraphSemanticSimilarity()


GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")


@app.route("/api/predict_quality", methods=["POST"])
def predict_quality():
    data = request.get_json()

    code = data.get("code", "")
    language = data.get("language", "").lower()

    if not code or not language:
        return jsonify({"error": "Both 'code' and 'language' are required."}), 400

    # Extract features
    quality_features = extractor.extract_features_from_code(code, language)
    if not quality_features:
        return jsonify({"error": "Unsupported language or failed to extract features."}), 400

    # Manually map your extracted features to the model-required ones
    feature_dict = {
        'cyclomatic_complexity': quality_features.cyclomatic_complexity,
        'num_functions': quality_features.function_count,
        'num_classes': quality_features.class_count,
        'comment_ratio': quality_features.comment_ratio,
        'max_nesting_depth': quality_features.max_nesting_depth,
        'has_error_handling': int(quality_features.try_catch_count > 0),
        'uses_framework': int(len(quality_features.framework_indicators) > 0),
        'num_imports': 1,  # Could be improved with static parsing
        'avg_function_length': quality_features.avg_function_length,
        'has_recursion': 0,  # Optional to implement
        'max_line_length': 80,  # Optional to calculate
        'is_object_oriented': int(quality_features.class_count > 0),
        'uses_logging': 1  # Optional to improve
    }

    # Order and convert for model
    X = [[feature_dict[name] for name in feature_names]]
    prediction = model.predict(X)[0]

    return jsonify({
        "predicted_code_quality": float(prediction),
        "features_used": feature_dict
    })


@app.route("/api/github_repos", methods=["GET"])
def get_github_repos():
    username = request.args.get("username")
    if not username:
        return jsonify({"error": "GitHub username is required as a query param"}), 400

    url = f"https://api.github.com/users/{username}/repos"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        repos = response.json()

        filtered_repos = [
            {
                "name": repo["name"],
                "html_url": repo["html_url"],
                "language": repo["language"],
                "description": repo["description"]
            }
            for repo in repos
            if repo["language"] in ["Java", "JavaScript"]
        ]

        return jsonify({
            "username": username,
            "filtered_repos": filtered_repos
        })

    except requests.exceptions.HTTPError as err:
        return jsonify({"error": f"GitHub API error: {err.response.status_code} {err.response.reason}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/github_profile_score", methods=["GET"])
def analyze_github_profile():
    username = request.args.get("username")
    if not username:
        return jsonify({"error": "GitHub username required"}), 400

    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }

    extractor = CodeFeatureExtractor()
    repo_scores = []
    all_scores = []

    # Step 1: Get Java/JS repos
    repos_url = f"https://api.github.com/users/{username}/repos"
    repos_res = requests.get(repos_url, headers=headers).json()
    relevant_repos = [r for r in repos_res if r['language'] in ['Java', 'JavaScript']]

    for repo in relevant_repos:
        repo_name = repo['name']
        owner = repo['owner']['login']

        # Step 2: Get commits by user
        commits_url = f"https://api.github.com/repos/{owner}/{repo_name}/commits?author={username}"
        commits_res = requests.get(commits_url, headers=headers).json()

        file_scores = []
        for commit in commits_res:
            sha = commit['sha']
            commit_url = f"https://api.github.com/repos/{owner}/{repo_name}/commits/{sha}"
            commit_data = requests.get(commit_url, headers=headers).json()

            for file in commit_data.get('files', []):
                filename = file['filename']
                if not (filename.endswith(".java") or filename.endswith(".js")):
                    continue

                # Step 3: Get file content
                raw_url = f"https://raw.githubusercontent.com/{owner}/{repo_name}/{sha}/{filename}"
                raw_res = requests.get(raw_url)
                if raw_res.status_code != 200:
                    continue
                code = raw_res.text

                # Step 4: Extract features and predict
                lang = "java" if filename.endswith(".java") else "javascript"
                features = extractor.extract_features_from_code(code, lang)

                if not features:
                    continue

                feature_dict = {
                    'cyclomatic_complexity': features.cyclomatic_complexity,
                    'num_functions': features.function_count,
                    'num_classes': features.class_count,
                    'comment_ratio': features.comment_ratio,
                    'max_nesting_depth': features.max_nesting_depth,
                    'has_error_handling': int(features.try_catch_count > 0),
                    'uses_framework': int(len(features.framework_indicators) > 0),
                    'num_imports': 1,
                    'avg_function_length': features.avg_function_length,
                    'has_recursion': 0,
                    'max_line_length': 80,
                    'is_object_oriented': int(features.class_count > 0),
                    'uses_logging': 1
                }

                X = [[feature_dict[name] for name in feature_names]]
                score = model.predict(X)[0]
                file_scores.append({ "path": filename, "score": float(score) })
                all_scores.append(score)

        if file_scores:
            avg = sum(f["score"] for f in file_scores) / len(file_scores)
            repo_scores.append({
                "repo": repo_name,
                "average_score": round(avg, 2),
                "files": file_scores
            })

    overall_score = round(sum(all_scores) / len(all_scores), 2) if all_scores else None

    return jsonify({
        "username": username,
        "overall_score": overall_score,
        "repo_scores": repo_scores
    })


@app.route('/api/github/<username>/algorithms', methods=['GET'])
def detect_user_algorithms(username):
    results = []
    repos = get_user_repos(username)

    for repo in repos:
        repo_name = repo.get("name")
        if not repo_name:
            continue

        repo_algos = set()
        code_files = get_java_js_files(username, repo_name)

        for file in code_files:
            code = fetch_code_from_url(file["raw_url"])
            language = file["language"]

            chunks = analyze_git_commit_chunk(code, language)
            for chunk in chunks:
                for match in chunk["detection_result"]["matches"]:
                    repo_algos.add(match["algorithm"])

        results.append({
            "repo": repo_name,
            "algorithms_detected": list(repo_algos)
        })

    return jsonify({
        "username": username,
        "algorithm_detection_summary": results
    })


@app.route('/api/coding-assessment/validate_code_answer', methods=['POST'])
def validate_code_answer():
    data = request.json
    language = data['language']
    candidate_code = data['candidate_code']
    reference_codes = data['reference_codes']

    result = compute_code_similarity(candidate_code, reference_codes, language)
    return jsonify(result)


@app.route('/api/coding-assessment/assess', methods=['POST'])
def assess():
    """
    Programming assessment endpoint - Java and JavaScript only
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data or 'language' not in data or 'candidate_code' not in data or 'reference_codes' not in data:
            return jsonify({"error": "Missing required fields: language, candidate_code, reference_codes"}), 400
        
        # Validate language support
        language = data['language'].lower()
        if language not in ['java', 'javascript']:
            return jsonify({"error": "Only Java and JavaScript are supported"}), 400
        
        if not isinstance(data['reference_codes'], list) or len(data['reference_codes']) == 0:
            return jsonify({"error": "reference_codes must be a non-empty list"}), 400
        
        # Perform assessment
        result = code_assessment_analyzer.assess(
            data['candidate_code'],
            data['reference_codes'],
            language
        )
        
        return jsonify(result)
    
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/generate_question", methods=["POST"])
def generate_coding_question():
    data = request.get_json()
    algo_summary = data.get("algorithm_detection_summary", [])
    experience_level = data.get("experience_level", "intern")
    language = data.get("language", "Java")  # default fallback

    output = []

    for repo_data in algo_summary:
        repo_name = repo_data.get("repo")
        algorithms = repo_data.get("algorithms_detected", [])

        # Infer language from repo name or metadata (you can improve this logic)
        repo_lang = "JavaScript" if "js" in repo_name.lower() else language

        for algo in algorithms:
            try:
                result = generate_question_and_refs(algo, experience_level, repo_lang)
                result["repo"] = repo_name
                result["algorithm"] = algo
                output.append(result)
            except Exception as e:
                output.append({
                    "repo": repo_name,
                    "algorithm": algo,
                    "error": str(e)
                })

    return jsonify({
        "username": data.get("username", ""),
        "generated_questions": output
    })

if __name__ == "__main__":
    app.run(debug=True)
   

