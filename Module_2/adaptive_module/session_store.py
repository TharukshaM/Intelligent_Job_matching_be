import uuid

# Global in-memory store
SESSIONS = {}

def create_session(experience, skills):
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {
        "experience": experience,
        "experience_encoded": None,
        "skills": skills,
        "current_complexity": None,
        "asked_questions": [],
        "score_history": []
    }
    return session_id

def get_session(session_id):
    return SESSIONS.get(session_id)

def update_session(session_id, updates):
    if session_id in SESSIONS:
        SESSIONS[session_id].update(updates)
