from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import esprima
import javalang
import re
import joblib


@dataclass
class CodeQualityFeatures:
    """Data class to hold extracted code quality features"""
    # Basic metrics
    lines_of_code: int
    blank_lines: int
    comment_lines: int

    # Complexity metrics
    cyclomatic_complexity: float
    function_count: int
    class_count: int
    max_nesting_depth: int

    # Maintainability metrics
    avg_function_length: float
    comment_ratio: float
    duplicate_code_ratio: float

    # Documentation metrics
    docstring_coverage: float

    # Consistency metrics
    naming_consistency_score: float

    # Error handling metrics
    try_catch_count: int

    # Language-specific features
    language: str
    framework_indicators: List[str]


class BaseCodeAnalyzer:
    """Base class for code analysis"""

    def __init__(self):
        self.features = {}

    def extract_basic_metrics(self, code: str) -> Dict[str, Any]:
        """Extract basic metrics common to all languages"""
        lines = code.split('\n')

        # Count different types of lines
        total_lines = len(lines)
        blank_lines = len([line for line in lines if line.strip() == ''])
        comment_lines = self.count_comment_lines(code)

        return {
            'lines_of_code': total_lines,
            'blank_lines': blank_lines,
            'comment_lines': comment_lines,
            'comment_ratio': comment_lines / max(total_lines, 1)
        }

    def count_comment_lines(self, code: str) -> int:
        """Count comment lines - to be implemented by subclasses"""
        raise NotImplementedError


class JavaScriptAnalyzer(BaseCodeAnalyzer):
    """Enhanced analyzer for JavaScript code"""

    def __init__(self):
        super().__init__()
        self.framework_indicators = {
            'react': ['React', 'useState', 'useEffect', 'Component', 'jsx'],
            'angular': ['@Component', '@Injectable', 'Observable', 'ngOnInit'],
            'vue': ['Vue', 'mounted', 'created', 'methods', 'computed'],
            'node': ['require', 'module.exports', 'process.env', 'fs.readFile'],
            'express': ['express', 'app.get', 'app.post', 'middleware']
        }

    def extract_features(self, code: str) -> CodeQualityFeatures:
        """Extract all features from JavaScript code"""
        try:
            # Extract basic metrics
            basic_metrics = self.extract_basic_metrics(code)

            # Extract complexity metrics using AST when possible
            if esprima:
                try:
                    parsed = esprima.parseScript(code, options={'tolerant': True})
                    complexity_metrics = self.extract_complexity_metrics_ast(parsed, code)
                except:
                    complexity_metrics = self.extract_complexity_metrics_regex(code)
            else:
                complexity_metrics = self.extract_complexity_metrics_regex(code)

            # Extract other metrics
            maintainability_metrics = self.extract_maintainability_metrics(code)
            doc_metrics = self.extract_documentation_metrics(code)
            consistency_metrics = self.extract_consistency_metrics(code)
            frameworks = self.detect_frameworks(code)

            return CodeQualityFeatures(
                # Basic metrics
                lines_of_code=basic_metrics['lines_of_code'],
                blank_lines=basic_metrics['blank_lines'],
                comment_lines=basic_metrics['comment_lines'],

                # Complexity metrics
                cyclomatic_complexity=complexity_metrics['cyclomatic_complexity'],
                function_count=complexity_metrics['function_count'],
                class_count=complexity_metrics['class_count'],
                max_nesting_depth=complexity_metrics['max_nesting_depth'],

                # Maintainability metrics
                avg_function_length=maintainability_metrics['avg_function_length'],
                comment_ratio=basic_metrics['comment_ratio'],
                duplicate_code_ratio=maintainability_metrics['duplicate_code_ratio'],

                # Documentation metrics
                docstring_coverage=doc_metrics['docstring_coverage'],

                # Consistency metrics
                naming_consistency_score=consistency_metrics['naming_consistency_score'],

                # Error handling metrics
                try_catch_count=complexity_metrics['try_catch_count'],

                # Language-specific
                language='javascript',
                framework_indicators=frameworks
            )

        except Exception as e:
            print(f"Error analyzing JavaScript code: {e}")
            return self.get_default_features('javascript')

    def count_comment_lines(self, code: str) -> int:
        """Count JavaScript comment lines"""
        comment_count = 0
        lines = code.split('\n')

        in_block_comment = False
        for line in lines:
            stripped = line.strip()

            if '/*' in stripped:
                in_block_comment = True
                comment_count += 1
                if '*/' in stripped:
                    in_block_comment = False
                continue

            if in_block_comment:
                comment_count += 1
                if '*/' in stripped:
                    in_block_comment = False
                continue

            if stripped.startswith('//'):
                comment_count += 1

        return comment_count

    def extract_complexity_metrics_ast(self, parsed_ast, code: str) -> Dict[str, Any]:
        """Extract complexity metrics using AST"""
        function_count = 0
        class_count = 0
        try_catch_count = 0
        complexity_sum = 0

        def walk_ast(node, depth=0):
            nonlocal function_count, class_count, try_catch_count, complexity_sum

            if isinstance(node, dict):
                node_type = node.get('type', '')

                if node_type in ['FunctionDeclaration', 'FunctionExpression', 'ArrowFunctionExpression']:
                    function_count += 1
                    complexity_sum += self.calculate_function_complexity(node)
                elif node_type == 'ClassDeclaration':
                    class_count += 1
                elif node_type == 'TryStatement':
                    try_catch_count += 1

                for key, value in node.items():
                    if isinstance(value, list):
                        for item in value:
                            walk_ast(item, depth + 1)
                    elif isinstance(value, dict):
                        walk_ast(value, depth + 1)

        walk_ast(parsed_ast)

        return {
            'cyclomatic_complexity': complexity_sum / max(function_count, 1),
            'function_count': function_count,
            'class_count': class_count,
            'max_nesting_depth': self.estimate_nesting_depth(code),
            'try_catch_count': try_catch_count
        }
    
    def extract_complexity_metrics_regex(self, code: str) -> Dict[str, Any]:
        """Fallback JavaScript complexity extraction"""
        function_count = len(re.findall(r'function\s+\w+\s*\([^)]*\)\s*{', code))
        class_count = len(re.findall(r'class\s+\w+\s*{', code))
        try_catch_count = len(re.findall(r'try\s*{', code))

        decision_points = len(re.findall(r'\b(if|else|for|while|switch|case)\b', code))
        complexity = decision_points / max(function_count, 1) + 1

        return {
            'cyclomatic_complexity': complexity,
            'function_count': function_count,
            'class_count': class_count,
            'max_nesting_depth': self.estimate_nesting_depth(code),
            'try_catch_count': try_catch_count
        }
    
    def get_default_features(self, language: str) -> CodeQualityFeatures:
        """Return default features when parsing fails"""
        return CodeQualityFeatures(
            lines_of_code=0, blank_lines=0, comment_lines=0,
            cyclomatic_complexity=1.0, function_count=0, class_count=0,
            max_nesting_depth=0, avg_function_length=0.0, comment_ratio=0.0,
            duplicate_code_ratio=0.0, docstring_coverage=0.0,
            naming_consistency_score=0.0, try_catch_count=0,
            language=language, framework_indicators=[]
        )
    
    def estimate_nesting_depth(self, code: str) -> int:
        """Estimate JavaScript nesting depth using curly brace count."""
        max_depth = 0
        current_depth = 0

        for line in code.splitlines():
            stripped = line.strip()
            current_depth += stripped.count('{')
            max_depth = max(max_depth, current_depth)
            current_depth -= stripped.count('}')

        return max_depth
    
    def extract_maintainability_metrics(self, code: str) -> Dict[str, Any]:
        """Extract JavaScript maintainability metrics"""
        lines = code.split('\n')
        function_starts = []

        # Find function declarations
        for i, line in enumerate(lines):
            if re.search(r'function\s+\w+\s*\([^)]*\)\s*{', line) or \
               re.search(r'\w+\s*=\s*function\s*\([^)]*\)\s*{', line) or \
               re.search(r'\w+\s*=>\s*{', line):
                function_starts.append(i)

        function_lengths = []
        if function_starts:
            for i, start in enumerate(function_starts):
                end = function_starts[i + 1] if i + 1 < len(function_starts) else len(lines)
                function_lengths.append(end - start)

        avg_function_length = sum(function_lengths) / max(len(function_lengths), 1)
        duplicate_ratio = self.estimate_duplicate_code(code)

        return {
            'avg_function_length': avg_function_length,
            'duplicate_code_ratio': duplicate_ratio
        }

    def extract_documentation_metrics(self, code: str) -> Dict[str, Any]:
        """Extract JavaScript documentation metrics"""
        # Count JSDoc comments
        jsdoc_count = len(re.findall(r'/\*\*[\s\S]*?\*/', code))
        
        # Count functions
        function_count = len(re.findall(r'function\s+\w+\s*\([^)]*\)\s*{', code))
        function_count += len(re.findall(r'\w+\s*=\s*function\s*\([^)]*\)\s*{', code))
        function_count += len(re.findall(r'\w+\s*=>\s*{', code))

        docstring_coverage = jsdoc_count / max(function_count, 1)

        return {
            'docstring_coverage': min(docstring_coverage, 1.0)
        }

    def extract_consistency_metrics(self, code: str) -> Dict[str, Any]:
        """Extract JavaScript consistency metrics"""
        identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code)

        if len(identifiers) < 2:
            return {'naming_consistency_score': 1.0}

        # JavaScript typically uses camelCase
        camelCase_count = sum(1 for id in identifiers if re.match(r'^[a-z][a-zA-Z0-9]*$', id))
        # Some constants use UPPER_CASE
        upper_case_count = sum(1 for id in identifiers if re.match(r'^[A-Z_]+$', id))

        total = len(identifiers)
        consistency = (camelCase_count + upper_case_count) / total

        return {
            'naming_consistency_score': min(consistency, 1.0)
        }

    def detect_frameworks(self, code: str) -> List[str]:
        """Detect JavaScript frameworks"""
        detected = []
        for framework, indicators in self.framework_indicators.items():
            for indicator in indicators:
                if indicator in code:
                    detected.append(framework)
                    break
        return detected

    def calculate_function_complexity(self, function_node) -> int:
        """Calculate JavaScript function complexity"""
        # This is a simplified version for AST-based analysis
        # In practice, you'd traverse the function node to count decision points
        complexity = 1
        
        # For now, return a basic complexity
        # You can enhance this based on your AST structure
        return complexity

    def estimate_duplicate_code(self, code: str) -> float:
        """Estimate JavaScript code duplication"""
        lines = [line.strip() for line in code.split('\n') if line.strip() and len(line.strip()) > 10]

        if len(lines) < 2:
            return 0.0

        duplicates = 0
        for i, line in enumerate(lines):
            for j in range(i + 1, len(lines)):
                if line == lines[j]:
                    duplicates += 1

        return duplicates / len(lines)



class JavaAnalyzer(BaseCodeAnalyzer):
    """Enhanced analyzer for Java code"""

    def __init__(self):
        super().__init__()
        self.framework_indicators = {
            'spring': ['@SpringBootApplication', '@RestController', '@Service', '@Repository', '@Autowired'],
            'hibernate': ['@Entity', '@Table', '@Column', '@OneToMany', '@ManyToOne'],
            'junit': ['@Test', '@Before', '@After', 'Assert.', 'assertTrue', 'assertEquals'],
        }

    def extract_features(self, code: str) -> CodeQualityFeatures:
        """Extract all features from Java code"""
        try:
            # Extract basic metrics
            basic_metrics = self.extract_basic_metrics(code)

            # Extract complexity metrics
            if javalang:
                try:
                    parsed = javalang.parse.parse(code)
                    complexity_metrics = self.extract_complexity_metrics_ast(parsed, code)
                except:
                    complexity_metrics = self.extract_complexity_metrics_regex(code)
            else:
                complexity_metrics = self.extract_complexity_metrics_regex(code)

            # Extract other metrics
            maintainability_metrics = self.extract_maintainability_metrics(code)
            doc_metrics = self.extract_documentation_metrics(code)
            consistency_metrics = self.extract_consistency_metrics(code)
            frameworks = self.detect_frameworks(code)

            return CodeQualityFeatures(
                lines_of_code=basic_metrics['lines_of_code'],
                blank_lines=basic_metrics['blank_lines'],
                comment_lines=basic_metrics['comment_lines'],
                cyclomatic_complexity=complexity_metrics['cyclomatic_complexity'],
                function_count=complexity_metrics['function_count'],
                class_count=complexity_metrics['class_count'],
                max_nesting_depth=complexity_metrics['max_nesting_depth'],
                avg_function_length=maintainability_metrics['avg_function_length'],
                comment_ratio=basic_metrics['comment_ratio'],
                duplicate_code_ratio=maintainability_metrics['duplicate_code_ratio'],
                docstring_coverage=doc_metrics['docstring_coverage'],
                naming_consistency_score=consistency_metrics['naming_consistency_score'],
                try_catch_count=complexity_metrics['try_catch_count'],
                language='java',
                framework_indicators=frameworks
            )

        except Exception as e:
            print(f"Error analyzing Java code: {e}")
            return self.get_default_features('java')

    def count_comment_lines(self, code: str) -> int:
        """Count Java comment lines"""
        comment_count = 0
        lines = code.split('\n')

        in_block_comment = False
        for line in lines:
            stripped = line.strip()

            if '/*' in stripped:
                in_block_comment = True
                comment_count += 1
                if '*/' in stripped:
                    in_block_comment = False
                continue

            if in_block_comment:
                comment_count += 1
                if '*/' in stripped:
                    in_block_comment = False
                continue

            if stripped.startswith('//'):
                comment_count += 1

        return comment_count

    def extract_complexity_metrics_ast(self, parsed_ast, code: str) -> Dict[str, Any]:
        """Extract complexity using Java AST"""
        method_count = 0
        class_count = 0
        try_catch_count = 0
        total_complexity = 0

        for path, node in parsed_ast.filter(javalang.tree.ClassDeclaration):
            class_count += 1

        for path, node in parsed_ast.filter(javalang.tree.MethodDeclaration):
            method_count += 1
            method_complexity = self.calculate_method_complexity(node)
            total_complexity += method_complexity

        for path, node in parsed_ast.filter(javalang.tree.TryStatement):
            try_catch_count += 1

        avg_complexity = total_complexity / max(method_count, 1)

        return {
            'cyclomatic_complexity': avg_complexity,
            'function_count': method_count,
            'class_count': class_count,
            'max_nesting_depth': self.estimate_nesting_depth(code),
            'try_catch_count': try_catch_count
        }

    def extract_complexity_metrics_regex(self, code: str) -> Dict[str, Any]:
        """Fallback Java complexity extraction"""
        method_count = len(re.findall(r'(public|private|protected)\s+[\w<>[\],\s]+\s+\w+\s*\([^)]*\)\s*{', code))
        class_count = len(re.findall(r'(public|private)?\s*class\s+\w+', code))
        try_catch_count = len(re.findall(r'try\s*{', code))

        decision_points = len(re.findall(r'\b(if|else|for|while|switch|case)\b', code))
        complexity = decision_points / max(method_count, 1) + 1

        return {
            'cyclomatic_complexity': complexity,
            'function_count': method_count,
            'class_count': class_count,
            'max_nesting_depth': self.estimate_nesting_depth(code),
            'try_catch_count': try_catch_count
        }

    def calculate_method_complexity(self, method_node) -> int:
        """Calculate Java method complexity"""
        complexity = 1

        for path, node in method_node.filter(javalang.tree.IfStatement):
            complexity += 1
        for path, node in method_node.filter(javalang.tree.WhileStatement):
            complexity += 1
        for path, node in method_node.filter(javalang.tree.ForStatement):
            complexity += 1
        for path, node in method_node.filter(javalang.tree.SwitchStatement):
            complexity += 1
        for path, node in method_node.filter(javalang.tree.CatchClause):
            complexity += 1

        return complexity

    def extract_maintainability_metrics(self, code: str) -> Dict[str, Any]:
        """Extract Java maintainability metrics"""
        lines = code.split('\n')
        method_starts = []

        for i, line in enumerate(lines):
            if re.search(r'(public|private|protected)\s+[\w<>[\],\s]+\s+\w+\s*\([^)]*\)\s*{', line):
                method_starts.append(i)

        method_lengths = []
        if method_starts:
            for i, start in enumerate(method_starts):
                end = method_starts[i + 1] if i + 1 < len(method_starts) else len(lines)
                method_lengths.append(end - start)

        avg_method_length = sum(method_lengths) / max(len(method_lengths), 1)
        duplicate_ratio = self.estimate_duplicate_code(code)

        return {
            'avg_function_length': avg_method_length,
            'duplicate_code_ratio': duplicate_ratio
        }

    def extract_documentation_metrics(self, code: str) -> Dict[str, Any]:
        """Extract Java documentation metrics"""
        javadoc_count = len(re.findall(r'/\*\*[\s\S]*?\*/', code))
        method_count = len(re.findall(r'(public|private|protected)\s+[\w<>[\],\s]+\s+\w+\s*\([^)]*\)\s*{', code))

        docstring_coverage = javadoc_count / max(method_count, 1)

        return {
            'docstring_coverage': min(docstring_coverage, 1.0)
        }

    def extract_consistency_metrics(self, code: str) -> Dict[str, Any]:
        """Extract Java consistency metrics"""
        identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code)

        if len(identifiers) < 2:
            return {'naming_consistency_score': 1.0}

        camelCase_count = sum(1 for id in identifiers if re.match(r'^[a-z][a-zA-Z0-9]*$', id))
        PascalCase_count = sum(1 for id in identifiers if re.match(r'^[A-Z][a-zA-Z0-9]*$', id))

        total = len(identifiers)
        consistency = (camelCase_count + PascalCase_count) / total

        return {
            'naming_consistency_score': min(consistency, 1.0)
        }

    def detect_frameworks(self, code: str) -> List[str]:
        """Detect Java frameworks"""
        detected = []
        for framework, indicators in self.framework_indicators.items():
            for indicator in indicators:
                if indicator in code:
                    detected.append(framework)
                    break
        return detected

    def estimate_nesting_depth(self, code: str) -> int:
        """Estimate Java nesting depth"""
        max_depth = 0
        current_depth = 0

        for char in code:
            if char == '{':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == '}':
                current_depth -= 1

        return max_depth

    def estimate_duplicate_code(self, code: str) -> float:
        """Estimate Java code duplication"""
        lines = [line.strip() for line in code.split('\n') if line.strip() and len(line.strip()) > 10]

        if len(lines) < 2:
            return 0.0

        duplicates = 0
        for i, line in enumerate(lines):
            for j in range(i + 1, len(lines)):
                if line == lines[j]:
                    duplicates += 1

        return duplicates / len(lines)

    def get_default_features(self, language: str) -> CodeQualityFeatures:
        """Return default features when parsing fails"""
        return CodeQualityFeatures(
            lines_of_code=0, blank_lines=0, comment_lines=0,
            cyclomatic_complexity=1.0, function_count=0, class_count=0,
            max_nesting_depth=0, avg_function_length=0.0, comment_ratio=0.0,
            duplicate_code_ratio=0.0, docstring_coverage=0.0,
            naming_consistency_score=0.0, try_catch_count=0,
            language=language, framework_indicators=[]
        )


class CodeFeatureExtractor:
    """Main class for extracting features from code"""

    def __init__(self):
        self.js_analyzer = JavaScriptAnalyzer()
        self.java_analyzer = JavaAnalyzer()

    def extract_features_from_code(self, code: str, language: str) -> Optional[CodeQualityFeatures]:
        """Extract features from code string"""
        language = language.lower()

        if language == 'javascript':
            return self.js_analyzer.extract_features(code)
        elif language == 'java':
            return self.java_analyzer.extract_features(code)
        else:
            print(f"Unsupported language: {language}")
            return None