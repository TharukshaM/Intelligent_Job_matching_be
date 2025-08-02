from difflib import SequenceMatcher

def compute_lcs_similarity(seq1, seq2):
    matcher = SequenceMatcher(None, seq1, seq2)
    return matcher.ratio()

def compute_code_similarity(candidate, references, language):
    from extractors.ast_extractor import extract_ast_nodes

    cand_nodes = extract_ast_nodes(candidate, language)
    scores = []

    for ref in references:
        ref_nodes = extract_ast_nodes(ref, language)
        lcs_score = compute_lcs_similarity(cand_nodes, ref_nodes)

        # Optional: add token overlap or other metrics
        token_overlap = len(set(cand_nodes) & set(ref_nodes)) / max(1, len(set(cand_nodes) | set(ref_nodes)))

        combined_score = (0.7 * lcs_score + 0.3 * token_overlap)
        scores.append({
            "score": combined_score,
            "lcs": lcs_score,
            "token_overlap": token_overlap
        })

    best_match = max(enumerate(scores), key=lambda x: x[1]["score"])
    return {
        "score": round(best_match[1]["score"], 3),
        "matched_reference_index": best_match[0],
        "similarity_type": "Graph-based",
        "details": {
            "graph_match_score": round(best_match[1]["score"], 3),
            "token_overlap": round(best_match[1]["token_overlap"], 3),
            "node_sequence_lcs": round(best_match[1]["lcs"], 3)
        }
    }
