import subprocess
import os
import re

def generate_question_and_refs(algorithm, experience, language):
    script_path = os.path.join(os.path.dirname(__file__), "generate_question_with_refs.js")
    try:
        result = subprocess.run(
            ["node", script_path, algorithm, experience, language],
            capture_output=True, text=True, check=True, encoding="utf-8"
        )
        output = result.stdout.strip()

        # Debug: print GPT response
        print("===== RAW GPT OUTPUT =====")
        print(output)
        print("==========================")

        # Extract Q, A1, A2 using stricter parsing
        question_match = re.search(r"Q:\s*(.*?)\s*A1:", output, re.DOTALL)
        a1_match = re.search(r"A1:\s*```(?:\w+)?\s*(.*?)```", output, re.DOTALL)
        a2_match = re.search(r"A2:\s*```(?:\w+)?\s*(.*?)```", output, re.DOTALL)

        question = question_match.group(1).strip() if question_match else ""
        code1 = a1_match.group(1).strip() if a1_match else ""
        code2 = a2_match.group(1).strip() if a2_match else ""

        return {
            "language": language.lower(),
            "Question": question,
            "reference_codes": [code1, code2]
        }

    except subprocess.CalledProcessError as e:
        return {"error": f"GPT generation failed: {e.stderr.strip()}"}
    except Exception as e:
        return {"error": str(e)}
