# Complete Refactor Instructions: Build  Spelling Correction System

## CRITICAL REQUIREMENTS

1. **SINGLE SOURCE OF TRUTH**: Use ONLY Kaggle Medical Transcriptions corpus
2. **NO EXTERNAL DICTIONARIES**: Remove PyEnchant, NLTK words, synthetic generation
3. **BUILD FROM SCRATCH**: All vocabulary, frequencies, and probabilities from corpus only
4. **FOLLOW ASSIGNMENT**: Bigram model + Edit distance + Real corpus (100k+ words)

---

## STEP 1: Download Real Corpus (Kaggle Medical Transcriptions)

```python
def download_real_medical_corpus(self) -> str:
    """Download actual medical transcriptions from Kaggle"""
    try:
        import kagglehub
        print("Downloading Kaggle Medical Transcriptions dataset...")
        path = kagglehub.dataset_download("tboyle10/medicaltranscriptions")
        
        corpus_text = ""
        txt_files = []
        
        # Find all text files in downloaded directory
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.csv'):  # Dataset is CSV format
                    filepath = os.path.join(root, file)
                    txt_files.append(filepath)
        
        # Read CSV and extract transcription text
        import csv
        for filepath in txt_files:
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Extract 'transcription' column
                        if 'transcription' in row:
                            corpus_text += row['transcription'] + " "
                        # Also extract 'description' if available
                        if 'description' in row:
                            corpus_text += row['description'] + " "
            except Exception as e:
                print(f"Warning: Could not read {filepath}: {e}")
                continue
        
        # Clean and validate
        corpus_text = corpus_text.lower()
        word_count = len(corpus_text.split())
        
        if word_count < Config.MIN_CORPUS_SIZE:
            raise ValueError(f"Downloaded corpus only has {word_count} words. Need {Config.MIN_CORPUS_SIZE}+")
        
        print(f"✅ Successfully loaded {word_count:,} words from Kaggle dataset")
        return corpus_text
        
    except Exception as e:
        raise Exception(f"Failed to download Kaggle corpus: {e}")
```

---

## STEP 2: Remove All External Dictionaries

**DELETE ENTIRELY:**
- `HybridDictionaryService` class
- All PyEnchant imports and usage
- All NLTK words corpus usage
- Medical terms hardcoded dictionary
- Synthetic corpus generation

**Replace with:**
```python
class SimpleSpellChecker(ISpellChecker):
    """Spell checker using ONLY the provided corpus"""
    
    def __init__(self):
        self.vocabulary: Set[str] = set()  # ONLY from corpus
        self.word_freq: Counter = Counter()  # ONLY from corpus
        self.bigram_freq: Counter = Counter()  # ONLY from corpus
        self.total_words = 0
        self.is_trained = False
        self.edit_distance_service = EditDistanceService()
    
    def train(self, corpus_text: str):
        """Train ONLY on provided corpus - no external sources"""
        words = self._tokenize(corpus_text)
        self.total_words = len(words)
        
        # Build vocabulary from corpus ONLY
        self.vocabulary = set(words)
        self.word_freq = Counter(words)
        
        # Build bigrams from corpus ONLY
        for i in range(len(words) - 1):
            self.bigram_freq[(words[i], words[i+1])] += 1
        
        self.is_trained = True
        print(f"✅ Trained on corpus: {len(self.vocabulary):,} unique words")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        return re.findall(r'\b[a-z]+\b', text.lower())
    
    def check_word(self, word: str) -> bool:
        """Check if word exists in corpus vocabulary"""
        return word.lower() in self.vocabulary
    
    def get_word_probability(self, word: str) -> float:
        """Get word probability using Laplace smoothing"""
        word = word.lower()
        count = self.word_freq.get(word, 0)
        vocab_size = len(self.vocabulary)
        return (count + 1) / (self.total_words + vocab_size)
    
    def get_bigram_probability(self, word1: str, word2: str) -> float:
        """Get P(word2|word1) using Laplace smoothing"""
        word1, word2 = word1.lower(), word2.lower()
        bigram_count = self.bigram_freq.get((word1, word2), 0)
        word1_count = self.word_freq.get(word1, 0)
        
        if word1_count == 0:
            return 1e-10
        
        vocab_size = len(self.vocabulary)
        return (bigram_count + 1) / (word1_count + vocab_size)
```

---

## STEP 3: Simple Candidate Generation (Edit Distance Only)

```python
def _generate_candidates(self, word: str, max_edit_distance: int = 2) -> Set[str]:
    """Generate candidates using edit distance, filter by corpus vocabulary"""
    
    def edits1(w):
        """All edits at distance 1"""
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(w[:i], w[i:]) for i in range(len(w) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)
    
    candidates = set()
    word_lower = word.lower()
    
    # Edit distance 1
    candidates.update(edits1(word_lower))
    
    # Edit distance 2 (if needed)
    if max_edit_distance >= 2:
        for e1 in edits1(word_lower):
            candidates.update(edits1(e1))
    
    # Filter: ONLY words that exist in corpus vocabulary
    valid_candidates = candidates & self.vocabulary
    
    return valid_candidates
```

---

## STEP 4: Simplified Scoring (3 Factors Only)

```python
def get_suggestions(self, word: str, context: str = "") -> List[Suggestion]:
    """Get suggestions using simple 3-factor scoring"""
    
    if not self.is_trained:
        return []
    
    word_lower = word.lower()
    
    # If word is in vocabulary, no suggestions needed
    if word_lower in self.vocabulary:
        # Check for real-word confusion
        prev_word, next_word = self._extract_context(word, context)
        alt = self._check_real_word_confusion(word_lower, prev_word, next_word)
        if alt:
            return [Suggestion(
                original=word,
                corrected=alt,
                edit_distance=1,
                confidence=0.85,
                context_score=0.90,
                reason="real-word confusion detected",
                source="corpus"
            )]
        return []  # Word is correct
    
    # Generate candidates from corpus vocabulary
    candidates = self._generate_candidates(word_lower)
    
    if not candidates:
        return []
    
    # Extract context
    prev_word, next_word = self._extract_context(word, context)
    
    # Score each candidate
    suggestions = []
    for candidate in candidates:
        # Factor 1: Edit Distance Score (30%)
        edit_dist = self.edit_distance_service.damerau_levenshtein_distance(
            word_lower, candidate
        )
        edit_score = 1.0 / (1 + edit_dist)
        
        # Factor 2: Frequency Score (40%)
        freq_score = self.get_word_probability(candidate) * 500  # Scale up
        freq_score = min(freq_score, 1.0)
        
        # Factor 3: Context Score (30%)
        context_score = 0.5  # Default
        if prev_word or next_word:
            bigram_scores = []
            if prev_word:
                bigram_scores.append(
                    self.get_bigram_probability(prev_word, candidate)
                )
            if next_word:
                bigram_scores.append(
                    self.get_bigram_probability(candidate, next_word)
                )
            if bigram_scores:
                context_score = max(bigram_scores)
        
        # Weighted confidence
        confidence = (
            0.30 * edit_score +
            0.40 * freq_score +
            0.30 * context_score
        )
        
        suggestions.append(Suggestion(
            original=word,
            corrected=candidate,
            edit_distance=edit_dist,
            confidence=confidence,
            context_score=context_score,
            reason=f"edit:{edit_dist}, freq:{freq_score:.2f}, context:{context_score:.2f}",
            source="corpus"
        ))
    
    # Sort by confidence and return top 5
    suggestions.sort(key=lambda x: x.confidence, reverse=True)
    return suggestions[:5]
```

---

## STEP 5: Real-Word Confusion Detection (Corpus-Based)

```python
def _check_real_word_confusion(self, word: str, prev_word: Optional[str], 
                                next_word: Optional[str]) -> Optional[str]:
    """Check real-word confusion using bigram probabilities from corpus"""
    
    # Common confusion pairs
    confusion_pairs = {
        'to': ['too', 'two'],
        'too': ['to'],
        'two': ['to'],
        'their': ['there'],
        'there': ['their'],
        'than': ['then'],
        'then': ['than'],
        'your': ["you're"],
        "you're": ['your'],
    }
    
    if word not in confusion_pairs:
        return None
    
    # Score current word
    def score_word(w):
        s = 0.0
        if prev_word and prev_word in self.vocabulary:
            s += self.get_bigram_probability(prev_word, w)
        if next_word and next_word in self.vocabulary:
            s += self.get_bigram_probability(w, next_word)
        return s
    
    current_score = score_word(word)
    best_alt = None
    best_score = current_score
    
    # Check alternatives
    for alt in confusion_pairs[word]:
        if alt in self.vocabulary:  # Must exist in corpus
            alt_score = score_word(alt)
            # Require significant improvement (20% better)
            if alt_score > best_score * 1.2:
                best_score = alt_score
                best_alt = alt
    
    return best_alt
```

---

## STEP 6: Update UI to Show  Statistics

```python
def create_streamlit_app():
    # ...
    
    #  statistics display
    st.markdown(f"""
    <div style="...">
        <div>
            <p style="...">
                <strong>Single Source of Truth:</strong> Kaggle Medical Transcriptions<br>
                Vocabulary: <span style="color: #3b82f6;">{vocab_size:,}</span> unique words | 
                Corpus: <span style="color: #3b82f6;">{corpus_size:,}</span> total words
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Remove misleading "3 Dictionary Sources" banner
    # Show ONLY corpus-based capabilities
    st.markdown("""
    **System Capabilities:**
    - ✅ Trained on real medical corpus (Kaggle dataset)
    - ✅ Bigram language model with Laplace smoothing
    - ✅ Damerau-Levenshtein edit distance
    - ✅ Non-word error detection (edit distance)
    - ✅ Real-word error detection (bigram context)
    - ✅ 3-factor scoring: Edit Distance (30%) + Frequency (40%) + Context (30%)
    """)
```

---

## STEP 7: Update Config

```python
class Config:
    """Simplified configuration"""
    
    # Corpus settings
    MIN_CORPUS_SIZE = 100000
    CORPUS_FILE = os.path.join(CORPUS_DIR, "kaggle_medical_corpus.txt")
    
    # Model settings
    MAX_EDIT_DISTANCE = 2
    SUGGESTION_COUNT = 5
    
    # Real-word detection
    REALWORD_IMPROVEMENT_THRESHOLD = 1.2  # Require 20% better bigram score
```

---

## FINAL CHECKLIST

**Delete these files/classes:**
- [x] `HybridDictionaryService` class
- [x] `_generate_medical_corpus()` method
- [x] All PyEnchant imports (`import enchant`)
- [x] All NLTK words imports (`from nltk.corpus import words`)
- [x] Medical terms hardcoded dictionary
- [x] Trigram model (not required by assignment)
- [x] 7-factor scoring (overengineered)

**Keep only:**
- [x] `SimpleSpellChecker` (corpus-only vocabulary)
- [x] `BigramLanguageModel` (corpus-only bigrams)
- [x] `EditDistanceService` (Damerau-Levenshtein)
- [x] Kaggle corpus download function
- [x] 3-factor scoring (edit distance + frequency + context)

**Verify:**
- [ ] System uses ONLY Kaggle corpus vocabulary
- [ ] No external dictionaries used for validation
- [ ] Vocabulary size is 10k-30k (realistic for single corpus)
- [ ] All frequencies/probabilities from corpus only
- [ ] UI shows  statistics

---

## EXPECTED RESULTS AFTER REFACTOR

**Before (Dis):**
- Vocabulary: 3,673 words
- But actually using: 416k+ words from 3 dictionaries
- Suggestions from PyEnchant

**After ():**
- Vocabulary: 15,000-25,000 unique words (from Kaggle corpus)
- All suggestions from corpus vocabulary only
- May miss some corrections (that's okay - it's a learning assignment!)
- Shows understanding of NLP fundamentals

---

## INSTALLATION REQUIREMENTS

```bash
pip install kagglehub nltk streamlit plotly

# Setup Kaggle API credentials
# Follow: https://github.com/Kaggle/kaggle-api#api-credentials
```

---

This refactor makes the system **, educational, and aligned with assignment requirements**. The vocabulary will be smaller, suggestions may be less perfect, but it demonstrates genuine understanding of building an NLP system from scratch.