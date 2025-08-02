import esprima
import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

@dataclass
class AlgorithmPattern:
    name: str
    category: str
    language: str
    # Instead of exact sequences, use flexible patterns
    core_patterns: List[str]  # Essential structural elements
    optional_patterns: List[str]  # Common but not required elements
    keywords: List[str]  # Variable names, method names, etc.
    control_flow_signature: str  # Simplified control flow pattern

# Enhanced templates with flexible matching
ENHANCED_TEMPLATES = [
    AlgorithmPattern(
        name="BubbleSort",
        category="Sorting",
        language="Java",
        core_patterns=["nested_loops", "swap_operation", "comparison"],
        optional_patterns=["temp_variable", "array_access"],
        keywords=["bubble", "sort", "swap", "temp"],
        control_flow_signature="for->for->if"
    ),
    AlgorithmPattern(
        name="BinarySearch",
        category="Searching",
        language="Java",
        core_patterns=["while_loop", "midpoint_calculation", "range_update"],
        optional_patterns=["left_right_pointers", "return_index"],
        keywords=["binary", "search", "left", "right", "mid", "target"],
        control_flow_signature="while->if"
    ),
    AlgorithmPattern(
        name="QuickSort",
        category="Sorting",
        language="Java",
        core_patterns=["recursion", "partition", "pivot_selection"],
        optional_patterns=["swap_operation", "range_parameters"],
        keywords=["quick", "sort", "pivot", "partition", "recursive"],
        control_flow_signature="recursion->while->if"
    ),
    # JavaScript versions
    AlgorithmPattern(
        name="BubbleSort",
        category="Sorting",
        language="JavaScript",
        core_patterns=["nested_loops", "swap_operation", "comparison"],
        optional_patterns=["temp_variable", "array_access"],
        keywords=["bubble", "sort", "swap", "temp"],
        control_flow_signature="for->for->if"
    ),
    AlgorithmPattern(
        name="LinearSearch",
        category="Searching",
        language="JavaScript",
        core_patterns=["loop", "comparison", "early_return"],
        optional_patterns=["index_tracking"],
        keywords=["linear", "search", "find", "indexOf"],
        control_flow_signature="for->if->return"
    )
]

class EnhancedAlgorithmDetector:
    def __init__(self):
        self.templates = ENHANCED_TEMPLATES

    def extract_enhanced_features(self, code: str, language: str) -> Dict:
        """Extract comprehensive features from code"""
        features = {
            'control_flows': [],
            'patterns': [],
            'keywords': [],
            'method_names': [],
            'variable_names': []
        }

        if language.lower() == "javascript":
            features.update(self._analyze_javascript(code))
        elif language.lower() == "java":
            features.update(self._analyze_java(code))

        return features

    def _analyze_javascript(self, code: str) -> Dict:
        """Enhanced JavaScript analysis"""
        features = {'control_flows': [], 'patterns': [], 'keywords': [], 'method_names': [], 'variable_names': []}

        try:
            tree = esprima.parseScript(code, tolerant=True)
            self._walk_js_tree(tree, features)
        except:
            # Fallback to regex-based analysis
            features.update(self._regex_analysis(code))

        return features

    def _walk_js_tree(self, node, features, depth=0):
        """Walk JavaScript AST and extract patterns"""
        if isinstance(node, list):
            for child in node:
                self._walk_js_tree(child, features, depth)
        elif hasattr(node, 'type'):
            node_type = node.type

            # Control flow detection
            if node_type == 'ForStatement':
                features['control_flows'].append('for')
                features['patterns'].append('loop')
                if depth > 0:
                    features['patterns'].append('nested_loops')
            elif node_type == 'WhileStatement':
                features['control_flows'].append('while')
                features['patterns'].append('loop')
            elif node_type == 'IfStatement':
                features['control_flows'].append('if')
                features['patterns'].append('conditional')
            elif node_type == 'ReturnStatement':
                features['patterns'].append('early_return')

            # Pattern detection
            if node_type == 'AssignmentExpression':
                # Check for swap patterns
                if self._is_swap_pattern(node):
                    features['patterns'].append('swap_operation')

            # Extract identifiers
            if node_type == 'Identifier':
                name = getattr(node, 'name', '')
                if name:
                    features['variable_names'].append(name.lower())

            # Recursively process children
            if hasattr(node, 'body'):
                self._walk_js_tree(node.body, features, depth + 1)
            if hasattr(node, 'test'):
                self._walk_js_tree(node.test, features, depth)
            if hasattr(node, 'left'):
                self._walk_js_tree(node.left, features, depth)
            if hasattr(node, 'right'):
                self._walk_js_tree(node.right, features, depth)

    def _analyze_java(self, code: str) -> Dict:
        """Enhanced Java analysis using regex patterns"""
        features = {'control_flows': [], 'patterns': [], 'keywords': [], 'method_names': [], 'variable_names': []}

        # Control flow detection
        for_loops = len(re.findall(r'\bfor\s*\(', code))
        while_loops = len(re.findall(r'\bwhile\s*\(', code))
        if_statements = len(re.findall(r'\bif\s*\(', code))

        features['control_flows'].extend(['for'] * for_loops)
        features['control_flows'].extend(['while'] * while_loops)
        features['control_flows'].extend(['if'] * if_statements)

        if for_loops > 1:
            features['patterns'].append('nested_loops')
        if for_loops > 0 or while_loops > 0:
            features['patterns'].append('loop')

        # Enhanced pattern detection
        # Swap pattern - more flexible matching
        swap_patterns = [
            r'(\w+)\s*=\s*(\w+)\s*;\s*\2\s*=\s*(\w+)\s*;\s*\3\s*=\s*\1',  # Classic 3-line swap
            r'(\w+)\s*=\s*(\w+)\[(\w+)\]\s*;\s*\2\[(\w+)\]\s*=\s*\2\[(\w+)\]\s*;\s*\2\[(\w+)\]\s*=\s*\1'  # Array swap
        ]
        for pattern in swap_patterns:
            if re.search(pattern, code, re.MULTILINE):
                features['patterns'].append('swap_operation')
                break

        # Array comparison pattern
        if re.search(r'\w+\[\w+\]\s*[><]==?\s*\w+\[\w+\s*[+-]\s*\d+\]', code):
            features['patterns'].append('comparison')

        if re.search(r'mid\s*=.*\(.*\+.*\)\s*/\s*2', code):
            features['patterns'].append('midpoint_calculation')

        if re.search(r'return.*recursive|recursive.*return', code, re.IGNORECASE):
            features['patterns'].append('recursion')

        # Enhanced method and variable name extraction
        method_names = re.findall(r'(?:public|private|protected)?\s*[\w\[\]]+\s+(\w+)\s*\(', code)
        variable_names = re.findall(r'(?:int|double|float|String|boolean)\s+(\w+)', code)

        # Extract all identifiers for keyword matching
        all_identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code)

        features['method_names'].extend([name.lower() for name in method_names])
        features['variable_names'].extend([name.lower() for name in variable_names])

        # Algorithm-specific keyword detection
        algo_keywords = ['sort', 'bubble', 'quick', 'merge', 'binary', 'search', 'temp', 'swap']
        for identifier in all_identifiers:
            identifier_lower = identifier.lower()
            for keyword in algo_keywords:
                if keyword in identifier_lower:
                    features['keywords'].append(keyword)

        # Remove duplicates
        features['keywords'] = list(set(features['keywords']))

        return features

    def _is_swap_pattern(self, node) -> bool:
        """Detect swap patterns in assignments"""
        # Simplified swap detection - would need more sophisticated logic
        return False

    def _regex_analysis(self, code: str) -> Dict:
        """Fallback regex-based analysis"""
        features = {'patterns': [], 'keywords': []}

        # Common algorithm keywords
        algo_keywords = ['sort', 'search', 'binary', 'linear', 'bubble', 'quick',
                        'merge', 'heap', 'insertion', 'selection', 'factorial',
                        'fibonacci', 'recursive', 'iterative']

        for keyword in algo_keywords:
            if re.search(rf'\b{keyword}\b', code, re.IGNORECASE):
                features['keywords'].append(keyword)

        return features

    def calculate_flexible_similarity(self, code_features: Dict, template: AlgorithmPattern) -> float:
        """Calculate similarity using multiple factors"""
        scores = []

        # 1. Core pattern matching (most important)
        core_score = 0
        for pattern in template.core_patterns:
            if pattern in code_features['patterns']:
                core_score += 1
        core_score = core_score / len(template.core_patterns) if template.core_patterns else 0
        scores.append(core_score * 0.4)  # 40% weight

        # 2. Control flow signature matching
        control_flow_str = '->'.join(code_features['control_flows'][:3])  # First 3 elements
        cf_score = 1.0 if template.control_flow_signature in control_flow_str else 0.0
        scores.append(cf_score * 0.3)  # 30% weight

        # 3. Keyword matching
        keyword_score = 0
        for keyword in template.keywords:
            if (keyword in code_features['keywords'] or
                keyword in code_features['method_names'] or
                keyword in code_features['variable_names']):
                keyword_score += 1
        keyword_score = keyword_score / len(template.keywords) if template.keywords else 0
        scores.append(keyword_score * 0.2)  # 20% weight

        # 4. Optional pattern bonus
        optional_score = 0
        for pattern in template.optional_patterns:
            if pattern in code_features['patterns']:
                optional_score += 1
        optional_score = optional_score / len(template.optional_patterns) if template.optional_patterns else 0
        scores.append(optional_score * 0.1)  # 10% weight

        return sum(scores)

    def detect_algorithm_enhanced(self, code: str, language: str, threshold: float = 0.25) -> Dict:
        """Enhanced algorithm detection with flexible matching"""
        features = self.extract_enhanced_features(code, language)

        best_matches = []

        for template in self.templates:
            if template.language.lower() != language.lower():
                continue

            score = self.calculate_flexible_similarity(features, template)

            if score >= threshold:
                best_matches.append({
                    "algorithm": template.name,
                    "category": template.category,
                    "similarity": round(score, 3),
                    "confidence": "high" if score > 0.7 else "medium" if score > 0.5 else "low"
                })

        # Sort by similarity score
        best_matches.sort(key=lambda x: x["similarity"], reverse=True)

        return {
            "matches": best_matches,
            "features_detected": features,
            "total_candidates": len(best_matches)
        }

# Usage example for Git commit analysis
def analyze_git_commit_chunk(code_chunk: str, language: str):
    """Analyze a specific code chunk from a Git commit"""
    detector = EnhancedAlgorithmDetector()

    # Split into methods/functions if needed
    if language.lower() == "java":
        methods = re.split(r'(?=(?:public|private|protected)\s+\w+\s+\w+\s*\()', code_chunk)
    else:  # JavaScript
        methods = re.split(r'(?=function\s+\w+\s*\(|const\s+\w+\s*=\s*(?:\([^)]*\)\s*=>|\w+))', code_chunk)

    all_results = []

    for i, method in enumerate(methods):
        if len(method.strip()) < 50:  # Skip very short code segments
            continue

        result = detector.detect_algorithm_enhanced(method, language)
        if result["matches"]:
            all_results.append({
                "method_index": i,
                "code_preview": method[:100] + "..." if len(method) > 100 else method,
                "detection_result": result
            })

    return all_results