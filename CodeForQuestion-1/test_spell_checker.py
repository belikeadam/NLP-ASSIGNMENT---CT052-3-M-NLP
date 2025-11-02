import unittest
import re
import heapq
from collections import defaultdict, Counter
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Import only the classes we need for testing, avoiding GUI imports
@dataclass
class Word:
    """Represents a word in the corpus with its frequency"""
    text: str
    frequency: int

    def __hash__(self):
        return hash(self.text)

@dataclass
class Suggestion:
    """Represents a spelling correction suggestion"""
    original: str
    corrected: str
    edit_distance: int
    confidence: float
    context_score: float

    def __lt__(self, other):
        return self.confidence > other.confidence

class ILanguageModel(ABC):
    """Interface for language model operations"""

    @abstractmethod
    def get_word_probability(self, word: str) -> float:
        pass

    @abstractmethod
    def get_bigram_probability(self, word1: str, word2: str) -> float:
        pass

class ISpellChecker(ABC):
    """Interface for spell checking operations"""

    @abstractmethod
    def check_word(self, word: str) -> bool:
        pass

    @abstractmethod
    def get_suggestions(self, word: str, context: str) -> List[Suggestion]:
        pass

class BigramLanguageModel(ILanguageModel):
    """Handles probability calculations for words and bigrams"""

    def __init__(self):
        self.word_freq: Dict[str, int] = Counter()
        self.bigram_freq: Dict[Tuple[str, str], int] = Counter()
        self.total_words = 0
        self.vocabulary: Set[str] = set()

    def train(self, text: str):
        """Train the model on corpus text"""
        words = self._tokenize(text)
        self.total_words = len(words)

        # Count word frequencies
        self.word_freq.update(words)
        self.vocabulary = set(words)

        # Count bigram frequencies
        for i in range(len(words) - 1):
            self.bigram_freq[(words[i], words[i+1])] += 1

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        return re.findall(r'\b[a-z]+\b', text.lower())

    def get_word_probability(self, word: str) -> float:
        """Get probability of a word using Laplace smoothing"""
        word = word.lower()
        count = self.word_freq.get(word, 0)
        # Laplace smoothing: (count + 1) / (total + vocabulary_size)
        return (count + 1) / (self.total_words + len(self.vocabulary) + 1)

    def get_bigram_probability(self, word1: str, word2: str) -> float:
        """Get conditional probability P(word2|word1)"""
        word1, word2 = word1.lower(), word2.lower()
        bigram_count = self.bigram_freq.get((word1, word2), 0)
        word1_count = self.word_freq.get(word1, 0)

        if word1_count == 0:
            return 1e-10  # Very small probability

        # Laplace smoothing for bigrams
        return (bigram_count + 1) / (word1_count + len(self.vocabulary))

class EditDistanceService:
    """Handles all edit distance calculations"""

    @staticmethod
    def minimum_edit_distance(source: str, target: str) -> int:
        """
        Calculate Levenshtein distance using dynamic programming.
        Supports: insertion, deletion, substitution operations.
        """
        m, n = len(source), len(target)

        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if source[i-1] == target[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],      # Deletion
                        dp[i][j-1],      # Insertion
                        dp[i-1][j-1]     # Substitution
                    )

        return dp[m][n]

    @staticmethod
    def damerau_levenshtein_distance(source: str, target: str) -> int:
        """
        Enhanced edit distance supporting transposition (swap adjacent chars).
        Example: 'hte' -> 'the' (distance = 1, not 2)
        """
        len1, len2 = len(source), len(target)
        big_int = len1 + len2

        # Character frequency map
        char_map = {}

        # Create distance matrix
        H = {}
        H[-1, -1] = big_int

        for i in range(0, len1 + 1):
            H[i, -1] = big_int
            H[i, 0] = i
        for j in range(0, len2 + 1):
            H[-1, j] = big_int
            H[0, j] = j

        for i in range(1, len1 + 1):
            DB = 0
            for j in range(1, len2 + 1):
                k = char_map.get(target[j-1], 0)
                l = DB

                if source[i-1] == target[j-1]:
                    cost = 0
                    DB = j
                else:
                    cost = 1

                H[i, j] = min(
                    H[i-1, j] + 1,           # Insertion
                    H[i, j-1] + 1,           # Deletion
                    H[i-1, j-1] + cost,      # Substitution
                    H[k-1, l-1] + (i-k-1) + 1 + (j-l-1)  # Transposition
                )

            char_map[source[i-1]] = i

        return H[len1, len2]

class SmartSuggestionService:
    """Generates and ranks spelling suggestions"""

    def __init__(self, language_model: ILanguageModel, edit_distance_service: EditDistanceService):
        self.language_model = language_model
        self.edit_distance = edit_distance_service
        self.max_edit_distance = 2

    def get_auto_suggestions(self, word: str, context: str, vocabulary: Set[str],
                            top_n: int = 5) -> List[Suggestion]:
        """
        Generate ranked suggestions using:
        1. Edit distance (Damerau-Levenshtein)
        2. Word frequency
        3. Bigram context
        """
        word_lower = word.lower()

        # Get context words
        prev_word, next_word = self._extract_context(word, context)

        # Generate candidate words within edit distance
        candidates = self._generate_candidates(word_lower, vocabulary)

        # Score and rank candidates
        suggestions = []
        for candidate in candidates:
            edit_dist = self.edit_distance.damerau_levenshtein_distance(word_lower, candidate)

            # Calculate confidence score
            word_prob = self.language_model.get_word_probability(candidate)
            context_score = self._calculate_context_score(candidate, prev_word, next_word)

            # Weighted confidence: 40% edit distance + 30% frequency + 30% context
            confidence = (
                0.4 * (1 / (1 + edit_dist)) +
                0.3 * min(word_prob * 100, 1.0) +
                0.3 * context_score
            )

            suggestions.append(Suggestion(
                original=word,
                corrected=candidate,
                edit_distance=edit_dist,
                confidence=confidence,
                context_score=context_score
            ))

        # Sort by confidence and return top N
        suggestions.sort()
        return suggestions[:top_n]

    def _generate_candidates(self, word: str, vocabulary: Set[str]) -> Set[str]:
        """Generate candidate words within max edit distance"""
        candidates = set()

        # Edit distance 1 variations
        candidates.update(self._edits1(word))

        # Edit distance 2 variations (optional, for better coverage)
        if self.max_edit_distance >= 2:
            for edit1 in self._edits1(word):
                candidates.update(self._edits1(edit1))

        # Filter to only words in vocabulary
        return candidates & vocabulary

    def _edits1(self, word: str) -> Set[str]:
        """Generate all strings at edit distance 1"""
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]

        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]

        return set(deletes + transposes + replaces + inserts)

    def _extract_context(self, word: str, full_text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract previous and next words from context"""
        words = re.findall(r'\b\w+\b', full_text.lower())

        try:
            idx = words.index(word.lower())
            prev_word = words[idx - 1] if idx > 0 else None
            next_word = words[idx + 1] if idx < len(words) - 1 else None
            return prev_word, next_word
        except ValueError:
            return None, None

    def _calculate_context_score(self, word: str, prev_word: Optional[str],
                                 next_word: Optional[str]) -> float:
        """Calculate context score using bigram probabilities"""
        score = 0.5  # Baseline

        if prev_word:
            score = max(score, self.language_model.get_bigram_probability(prev_word, word))

        if next_word:
            score = max(score, self.language_model.get_bigram_probability(word, next_word))

        return min(score, 1.0)

class AdvancedSpellChecker(ISpellChecker):
    """Main spell checker combining all components"""

    def __init__(self):
        self.language_model = BigramLanguageModel()
        self.edit_distance_service = EditDistanceService()
        self.suggestion_service = SmartSuggestionService(
            self.language_model,
            self.edit_distance_service
        )
        self.vocabulary: Set[str] = set()
        self.is_trained = False

    def train(self, corpus: str, progress_callback=None):
        """Train the spell checker on a corpus"""
        if progress_callback:
            progress_callback("Training language model...")

        self.language_model.train(corpus)
        self.vocabulary = self.language_model.vocabulary
        self.is_trained = True

        if progress_callback:
            progress_callback(f"Training complete! Vocabulary: {len(self.vocabulary)} words")

    def check_word(self, word: str) -> bool:
        """Check if a word is spelled correctly"""
        return word.lower() in self.vocabulary

    def get_suggestions(self, word: str, context: str = "") -> List[Suggestion]:
        """Get spelling suggestions for a misspelled word"""
        if not self.is_trained:
            return []

        return self.suggestion_service.get_auto_suggestions(
            word, context, self.vocabulary
        )

    def get_all_words_sorted(self) -> List[Tuple[str, int]]:
        """Get sorted list of all words with frequencies"""
        word_freq = [(word, self.language_model.word_freq[word])
                     for word in self.vocabulary]
        return sorted(word_freq, key=lambda x: (-x[1], x[0]))

class TestEditDistance(unittest.TestCase):
    def setUp(self):
        self.service = EditDistanceService()

    def test_minimum_edit_distance(self):
        # Test basic operations
        self.assertEqual(self.service.minimum_edit_distance("cat", "hat"), 1)  # substitution
        self.assertEqual(self.service.minimum_edit_distance("cat", "cats"), 1)  # insertion
        self.assertEqual(self.service.minimum_edit_distance("cats", "cat"), 1)  # deletion

    def test_damerau_levenshtein(self):
        # Test transposition
        self.assertEqual(self.service.damerau_levenshtein_distance("hte", "the"), 1)
        self.assertEqual(self.service.damerau_levenshtein_distance("graffe", "giraffe"), 1)

class TestLanguageModel(unittest.TestCase):
    def setUp(self):
        self.model = BigramLanguageModel()
        corpus = "the cat sat on the mat the dog sat on the log"
        self.model.train(corpus)

    def test_word_probability(self):
        # Most frequent word should have higher probability
        prob_the = self.model.get_word_probability("the")
        prob_cat = self.model.get_word_probability("cat")
        self.assertGreater(prob_the, prob_cat)

    def test_bigram_probability(self):
        # "the cat" appears in corpus
        prob = self.model.get_bigram_probability("the", "cat")
        self.assertGreater(prob, 0)

class TestSpellChecker(unittest.TestCase):
    def setUp(self):
        self.checker = AdvancedSpellChecker()
        corpus = "the quick brown fox jumps over the lazy dog " * 100
        self.checker.train(corpus)

    def test_check_word(self):
        self.assertTrue(self.checker.check_word("the"))
        self.assertTrue(self.checker.check_word("quick"))
        self.assertFalse(self.checker.check_word("teh"))

    def test_get_suggestions(self):
        suggestions = self.checker.get_suggestions("teh", "the quick brown")
        self.assertTrue(len(suggestions) > 0)
        self.assertEqual(suggestions[0].corrected, "the")

if __name__ == "__main__":
    unittest.main()