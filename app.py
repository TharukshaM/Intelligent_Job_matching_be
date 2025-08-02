from flask import Flask, request, jsonify
from Module_2.softskill_module.evaluator import evaluate_communication, get_label, evaluate_answer_relevance, adjust_final_score_based_on_relevance
from Module_2.softskill_module.question_selector import api_get_question_by_experience
from Module_2.adaptive_module.adaptive_engine import (
    question_bank,
    get_initial_complexity, encode_experience,
    get_question_by_complexity, evaluate_answer, predict_next_complexity
)
from Module_2.adaptive_module.session_store import create_session, get_session, update_session


app = Flask(__name__)

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


if __name__ == "__main__":
    app.run(debug=True)
