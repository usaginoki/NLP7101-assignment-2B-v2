"""
Feature extraction modules for code analysis
"""

import re
from typing import Dict, Optional
import radon.complexity as radon_cc
import radon.metrics as radon_metrics
import radon.raw as radon_raw
from lexicalrichness import LexicalRichness
import warnings


class ComplexityFeatureExtractor:
    """Extract code complexity metrics using radon"""

    def __init__(self):
        pass

    def extract(self, code: str) -> Dict[str, float]:
        """
        Extract complexity features from code

        Args:
            code: Source code string

        Returns:
            Dictionary of complexity features
        """
        features = {}

        try:
            # Lines of Code metrics
            loc_metrics = self._extract_loc_metrics(code)
            features.update(loc_metrics)

            # Cyclomatic Complexity
            cc_metrics = self._extract_cyclomatic_complexity(code)
            features.update(cc_metrics)

            # Halstead metrics
            halstead_metrics = self._extract_halstead_metrics(code)
            features.update(halstead_metrics)

            # Maintainability Index
            mi = self._extract_maintainability_index(code)
            features['maintainability_index'] = mi

        except Exception as e:
            # Return default values on error
            features = self._get_default_features()
            features['extraction_error'] = 1

        return features

    def _extract_loc_metrics(self, code: str) -> Dict[str, float]:
        """Extract Lines of Code metrics"""
        try:
            analysis = radon_raw.analyze(code)
            return {
                'loc': analysis.loc,
                'lloc': analysis.lloc,
                'sloc': analysis.sloc,
                'comments': analysis.comments,
                'multi': analysis.multi,
                'blank': analysis.blank,
                'single_comments': analysis.single_comments,
            }
        except:
            return {
                'loc': 0, 'lloc': 0, 'sloc': 0,
                'comments': 0, 'multi': 0, 'blank': 0,
                'single_comments': 0
            }

    def _extract_cyclomatic_complexity(self, code: str) -> Dict[str, float]:
        """Extract cyclomatic complexity metrics"""
        try:
            cc_results = radon_cc.cc_visit(code)
            if cc_results:
                complexities = [block.complexity for block in cc_results]
                return {
                    'cc_max': max(complexities),
                    'cc_mean': sum(complexities) / len(complexities),
                    'cc_total': sum(complexities),
                    'cc_count': len(complexities),
                }
            else:
                return {'cc_max': 0, 'cc_mean': 0, 'cc_total': 0, 'cc_count': 0}
        except:
            return {'cc_max': 0, 'cc_mean': 0, 'cc_total': 0, 'cc_count': 0}

    def _extract_halstead_metrics(self, code: str) -> Dict[str, float]:
        """Extract Halstead complexity metrics"""
        try:
            h = radon_metrics.h_visit(code)
            if h and hasattr(h, 'total'):
                ht = h.total
                return {
                    'halstead_vocabulary': ht.vocabulary,
                    'halstead_length': ht.length,
                    'halstead_volume': ht.volume,
                    'halstead_difficulty': ht.difficulty,
                    'halstead_effort': ht.effort,
                    'halstead_bugs': ht.bugs,
                    'halstead_time': ht.time,
                }
            else:
                return self._get_default_halstead()
        except:
            return self._get_default_halstead()

    def _get_default_halstead(self) -> Dict[str, float]:
        """Return default Halstead metrics"""
        return {
            'halstead_vocabulary': 0,
            'halstead_length': 0,
            'halstead_volume': 0,
            'halstead_difficulty': 0,
            'halstead_effort': 0,
            'halstead_bugs': 0,
            'halstead_time': 0,
        }

    def _extract_maintainability_index(self, code: str) -> float:
        """Extract Maintainability Index"""
        try:
            mi = radon_metrics.mi_visit(code, multi=True)
            return mi if isinstance(mi, (int, float)) else 0
        except:
            return 0

    def _get_default_features(self) -> Dict[str, float]:
        """Return default feature values"""
        features = {
            'loc': 0, 'lloc': 0, 'sloc': 0,
            'comments': 0, 'multi': 0, 'blank': 0,
            'single_comments': 0,
            'cc_max': 0, 'cc_mean': 0, 'cc_total': 0, 'cc_count': 0,
            'maintainability_index': 0,
            'extraction_error': 0,
        }
        features.update(self._get_default_halstead())
        return features


class LexicalFeatureExtractor:
    """Extract lexical diversity and token-based features"""

    def __init__(self):
        # Common programming keywords (Python, Java, JavaScript, C++)
        self.keywords = set([
            'if', 'else', 'for', 'while', 'return', 'def', 'class',
            'import', 'from', 'try', 'except', 'finally', 'with',
            'void', 'int', 'float', 'double', 'char', 'string',
            'public', 'private', 'protected', 'static', 'final',
            'function', 'var', 'let', 'const', 'async', 'await',
        ])

    def extract(self, code: str) -> Dict[str, float]:
        """
        Extract lexical features from code

        Args:
            code: Source code string

        Returns:
            Dictionary of lexical features
        """
        features = {}

        try:
            # Basic token statistics
            token_features = self._extract_token_features(code)
            features.update(token_features)

            # Lexical diversity using lexicalrichness
            lex_features = self._extract_lexical_diversity(code)
            features.update(lex_features)

            # Keyword usage
            keyword_features = self._extract_keyword_features(code)
            features.update(keyword_features)

        except Exception as e:
            features = self._get_default_features()
            features['lex_extraction_error'] = 1

        return features

    def _extract_token_features(self, code: str) -> Dict[str, float]:
        """Extract basic token statistics"""
        # Tokenize (simple word boundary split)
        tokens = re.findall(r'\b\w+\b', code)

        if not tokens:
            return {
                'token_count': 0,
                'unique_tokens': 0,
                'avg_token_length': 0,
            }

        unique_tokens = set(tokens)

        return {
            'token_count': len(tokens),
            'unique_tokens': len(unique_tokens),
            'avg_token_length': sum(len(t) for t in tokens) / len(tokens),
        }

    def _extract_lexical_diversity(self, code: str) -> Dict[str, float]:
        """Extract lexical diversity metrics using lexicalrichness"""
        try:
            # Tokenize
            tokens = re.findall(r'\b\w+\b', code)
            text = ' '.join(tokens)

            if len(tokens) < 10:  # Too short for meaningful analysis
                return self._get_default_lexical_diversity()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lex = LexicalRichness(text)

                features = {
                    'ttr': lex.ttr,
                }

                # Try to compute MTLD (might fail on short texts)
                try:
                    features['mtld'] = lex.mtld(threshold=0.72)
                except:
                    features['mtld'] = 0

                # Try to compute MATTR
                try:
                    features['mattr'] = lex.mattr(window_size=min(50, len(tokens)))
                except:
                    features['mattr'] = 0

                return features

        except:
            return self._get_default_lexical_diversity()

    def _get_default_lexical_diversity(self) -> Dict[str, float]:
        """Return default lexical diversity values"""
        return {
            'ttr': 0,
            'mtld': 0,
            'mattr': 0,
        }

    def _extract_keyword_features(self, code: str) -> Dict[str, float]:
        """Extract programming keyword usage features"""
        tokens = re.findall(r'\b\w+\b', code.lower())

        if not tokens:
            return {
                'keyword_count': 0,
                'keyword_ratio': 0,
            }

        keyword_count = sum(1 for token in tokens if token in self.keywords)

        return {
            'keyword_count': keyword_count,
            'keyword_ratio': keyword_count / len(tokens) if tokens else 0,
        }

    def _get_default_features(self) -> Dict[str, float]:
        """Return default feature values"""
        features = {
            'token_count': 0,
            'unique_tokens': 0,
            'avg_token_length': 0,
            'keyword_count': 0,
            'keyword_ratio': 0,
            'lex_extraction_error': 0,
        }
        features.update(self._get_default_lexical_diversity())
        return features
