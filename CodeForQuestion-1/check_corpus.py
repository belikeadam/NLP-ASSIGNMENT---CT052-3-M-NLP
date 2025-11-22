import sys
sys.path.append(r"C:\Users\Administrator\Downloads\Assignment\NLP-ASSIGNMENT---CT052-3-M-NLP\CodeForQuestion-1")

# Delete cache first!
import os
import shutil
cache_dir = r"C:\Users\Administrator\Downloads\Assignment\NLP-ASSIGNMENT---CT052-3-M-NLP\CodeForQuestion-1\cache"
corpus_file = r"C:\Users\Administrator\Downloads\Assignment\NLP-ASSIGNMENT---CT052-3-M-NLP\CodeForQuestion-1\corpus\medical_corpus.txt"

if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print("✅ Cache deleted")

if os.path.exists(corpus_file):
    os.remove(corpus_file)
    print("✅ Old corpus deleted")

# Now test
from spell_correction_system import CorpusService, BigramLanguageModel, AdvancedSpellChecker

print("\n" + "="*70)
print("TESTING ENHANCED SPELLING CORRECTION")
print("="*70)

# Load corpus
corpus_service = CorpusService()
corpus = corpus_service.load_corpus()

# Train model
model = BigramLanguageModel()
model.train(corpus)

print(f"\n✅ Vocabulary: {len(model.vocabulary):,} unique words")
print(f"✅ Bigrams: {len(model.bigram_freq):,} pairs")

# Test real-word detection capability
test_cases = [
    ("to", "the", "admitted"),      # Correct: to the
    ("too", "the", "admitted"),     # Wrong: too the (should prefer "to")
    ("their", "family", None),      # Correct: their family
    ("there", "family", None),      # Wrong: there family (should prefer "their")
]

print(f"\n✅ Bigram Probability Tests (Real-Word Detection):")
for w1, w2, context in test_cases:
    prob = model.get_bigram_probability(w1, w2)
    print(f"   P({w2} | {w1}) = {prob:.6f}")

# Test spell checker
checker = AdvancedSpellChecker()
checker.language_model = model
checker.vocabulary = model.vocabulary
checker.is_trained = True

print(f"\n✅ Spelling Correction Tests:")
test_words = [
    ("pationt", "patient"),
    ("treatmant", "treatment"),
    ("diagnosi", "diagnosis"),
]

for wrong, expected in test_words:
    suggestions = checker.get_suggestions(wrong, "")
    if suggestions:
        top = suggestions[0].corrected
        print(f"   '{wrong}' → '{top}' (expected: '{expected}') {'✅' if top == expected else '❌'}")

print("\n" + "="*70)
print("✅ ENHANCED SYSTEM READY!")
print("="*70)