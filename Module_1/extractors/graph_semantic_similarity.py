import re
import difflib
import numpy as np
from collections import defaultdict


class GraphSemanticSimilarity:
    def __init__(self):
        # Optimal weights from the research paper (Section 3, Step 3)
        self.a = 0.6416  # entropy weight
        self.b = 0.3584  # LCS weight
        self.supported_languages = ['java', 'javascript']
    
    def standardize_code(self, code, language):
        """
        Code standardization for Java and JavaScript only
        Based on paper Section 3, Step 2.2
        """
        if language.lower() == 'java':
            return self._standardize_java(code)
        elif language.lower() == 'javascript':
            return self._standardize_javascript(code)
        else:
            raise ValueError(f"Unsupported language: {language}. Only Java and JavaScript are supported.")
    
    def _standardize_java(self, code):
        """Java-specific standardization"""
        # Remove Java comments
        code = re.sub(r'//.*|/\*.*?\*/', '', code, flags=re.DOTALL)
        # Normalize whitespace
        code = re.sub(r'\s+', ' ', code.strip())
        # Preserve Java keywords
        java_keywords = ['public', 'private', 'protected', 'static', 'final', 'class', 'interface', 
                        'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'return', 'void',
                        'int', 'String', 'boolean', 'double', 'float', 'long', 'char', 'new',
                        'this', 'super', 'extends', 'implements', 'import', 'package']
        
        # Replace identifiers with '#' but preserve keywords
        tokens = code.split()
        standardized_tokens = []
        for token in tokens:
            clean_token = re.sub(r'[{}();,\[\]]', '', token)
            if clean_token in java_keywords or re.match(r'^\d+$', clean_token):
                standardized_tokens.append(token)
            elif re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*', clean_token):
                standardized_tokens.append(re.sub(r'[a-zA-Z_][a-zA-Z0-9_]*', '#', token))
            else:
                standardized_tokens.append(token)
        
        return ' '.join(standardized_tokens)
    

    def _standardize_javascript(self, code):
        """JavaScript-specific standardization"""
        # Remove JavaScript comments
        code = re.sub(r'//.*|/\*.*?\*/', '', code, flags=re.DOTALL)
        # Normalize whitespace
        code = re.sub(r'\s+', ' ', code.strip())
        # Preserve JavaScript keywords
        js_keywords = ['function', 'var', 'let', 'const', 'if', 'else', 'for', 'while', 'do',
                    'switch', 'case', 'return', 'true', 'false', 'null', 'undefined',
                    'typeof', 'instanceof', 'new', 'this', 'try', 'catch', 'finally',
                    'class', 'extends', 'constructor', 'async', 'await', 'import', 'export',
                    'default', 'break', 'continue', 'throw', 'delete', 'in', 'of']
        
        # Replace identifiers with '#' but preserve keywords
        tokens = code.split()
        standardized_tokens = []
        for token in tokens:
            clean_token = re.sub(r'[{}();,\[\]]', '', token)
            if clean_token in js_keywords or re.match(r'^\d+$', clean_token):
                standardized_tokens.append(token)
            elif re.match(r'^[a-zA-Z_$][a-zA-Z0-9_$]*', clean_token):
                standardized_tokens.append(re.sub(r'[a-zA-Z_$][a-zA-Z0-9_$]*', '#', token))
            else:
                standardized_tokens.append(token)
        
        return ' '.join(standardized_tokens)
    
    def extract_semantic_tokens(self, standardized_code):
        """
        Extract semantic tokens for dependency graph analysis
        Based on paper's AST feature extraction
        """
        tokens = []
        # Split by common delimiters and filter empty strings
        raw_tokens = re.split(r'[{}();,\s]+', standardized_code)
        for token in raw_tokens:
            if token.strip():
                tokens.append(token.strip())
        return tokens
    
    def calculate_lcs_similarity(self, tokens1, tokens2):
        """
        Calculate LCS similarity (paper Section 3, Step 3.3)
        """
        if not tokens1 or not tokens2:
            return 0.0
        
        matcher = difflib.SequenceMatcher(None, tokens1, tokens2)
        lcs_length = sum(match.size for match in matcher.get_matching_blocks())
        
        # Paper formula: equation (9)
        return (2.0 * lcs_length) / (len(tokens1) + len(tokens2))
    
    def calculate_entropy_similarity(self, tokens1, tokens2):
        """
        Calculate relative entropy similarity (paper Section 3, Step 3.2)
        """
        if not tokens1 or not tokens2:
            return 0.0
        
        # Create probability distributions
        all_tokens = set(tokens1 + tokens2)
        
        def get_prob_dist(tokens):
            counts = defaultdict(int)
            for token in tokens:
                counts[token] += 1
            total = len(tokens)
            return {token: (counts.get(token, 0) + 1e-10) / (total + 1e-10) 
                   for token in all_tokens}
        
        prob1 = get_prob_dist(tokens1)
        prob2 = get_prob_dist(tokens2)
        
        # Calculate KL divergence (paper equation 6)
        kl1 = sum(prob1[token] * np.log(prob1[token] / prob2[token]) 
                 for token in all_tokens)
        kl2 = sum(prob2[token] * np.log(prob2[token] / prob1[token]) 
                 for token in all_tokens)
        
        # Convert to similarity (paper equation 8)
        similarity = 1 - (kl1 + kl2) / 2
        return max(0, min(1, similarity))
    
    def calculate_similarity(self, candidate_tokens, reference_tokens):
        """
        Calculate final similarity using weighted combination (paper equation 10)
        """
        lcs_sim = self.calculate_lcs_similarity(candidate_tokens, reference_tokens)
        entropy_sim = self.calculate_entropy_similarity(candidate_tokens, reference_tokens)
        
        # Paper's weighted combination: Sim(x,y) = a × SimKL(x,y) + b × SimLCS(x,y)
        return self.a * entropy_sim + self.b * lcs_sim
    
    def assess(self, candidate_code, reference_codes, language):
        """
        Main assessment function for Java and JavaScript only
        Following paper's 4-step algorithm
        """
        # Validate language
        if language.lower() not in self.supported_languages:
            raise ValueError(f"Unsupported language: {language}. Only {', '.join(self.supported_languages)} are supported.")
        
        # Step 2: Standardize candidate code
        candidate_standardized = self.standardize_code(candidate_code, language)
        candidate_tokens = self.extract_semantic_tokens(candidate_standardized)
        
        similarities = []
        
        # Step 3: Calculate similarity with each reference
        for ref_code in reference_codes:
            ref_standardized = self.standardize_code(ref_code, language)
            ref_tokens = self.extract_semantic_tokens(ref_standardized)
            
            similarity = self.calculate_similarity(candidate_tokens, ref_tokens)
            similarities.append(similarity)
        
        # Step 4: Scoring - use maximum similarity (paper equation 14)
        max_similarity = max(similarities) if similarities else 0
        final_score = max_similarity * 10  # Scale to 0-10
        
        return {
            "score": round(final_score, 2),
            "similarity": round(max_similarity, 4)
        }