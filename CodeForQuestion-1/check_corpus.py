"""
Diagnostic Tool: Verify Corpus Loading
Run this to check what's actually being loaded
"""

import os
import sys

# Add your script's directory to path
script_dir = r"C:\Users\Administrator\Downloads\Assignment\NLP-ASSIGNMENT---CT052-3-M-NLP\CodeForQuestion-1"
sys.path.insert(0, script_dir)

print("="*70)
print("CORPUS LOADING DIAGNOSTIC")
print("="*70)

# Check file existence
corpus_path = os.path.join(script_dir, "corpus", "medical_corpus.txt")
print(f"\n1. Checking file path: {corpus_path}")
print(f"   File exists: {os.path.exists(corpus_path)}")

if os.path.exists(corpus_path):
    # Read and analyze
    with open(corpus_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    words = content.split()
    unique_words = set(words)
    
    print(f"\n2. File Statistics:")
    print(f"   Total characters: {len(content):,}")
    print(f"   Total words: {len(words):,}")
    print(f"   Unique words: {len(unique_words):,}")
    
    print(f"\n3. First 200 characters:")
    print(f"   {content[:200]}...")
    
    print(f"\n4. Sample unique words:")
    sample_words = sorted(unique_words)[:50]
    print(f"   {', '.join(sample_words[:20])}")
    
    print(f"\n5. Content Analysis:")
    medical_keywords = ['patient', 'diagnosis', 'treatment', 'medical', 'clinical', 
                       'examination', 'therapy', 'syndrome', 'blood', 'surgery']
    found_keywords = [kw for kw in medical_keywords if kw in content.lower()]
    print(f"   Medical keywords found: {len(found_keywords)}/{len(medical_keywords)}")
    print(f"   Keywords: {', '.join(found_keywords)}")
    
    print(f"\n✅ CONCLUSION: Your corpus IS valid and WILL be used by the script!")
    
else:
    print("\n❌ ERROR: Corpus file not found!")
    print("   Expected location:", corpus_path)
    print("\n   Please verify:")
    print("   1. File actually exists at this path")
    print("   2. File name is exactly 'medical_corpus.txt'")
    print("   3. No permission issues")

print("\n" + "="*70)
print("CHECKING CACHE FILES")
print("="*70)

cache_dir = os.path.join(script_dir, "cache")
cache_file = os.path.join(cache_dir, "language_model.pkl")

print(f"\nCache directory: {cache_dir}")
print(f"Cache exists: {os.path.exists(cache_dir)}")

if os.path.exists(cache_file):
    import pickle
    try:
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        
        print(f"\n✅ Cached model found:")
        print(f"   Vocabulary size: {len(data['vocabulary']):,} words")
        print(f"   Total words processed: {data['total_words']:,}")
        print(f"   Bigrams stored: {len(data['bigram_freq']):,}")
        
        # Sample vocabulary
        sample_vocab = sorted(list(data['vocabulary']))[:20]
        print(f"\n   Sample vocabulary: {', '.join(sample_vocab)}")
        
    except Exception as e:
        print(f"❌ Error reading cache: {e}")
else:
    print("\n⚠️  No cached model found - will train from corpus on first run")

print("\n" + "="*70)

# Now test actual loading
print("\nTESTING ACTUAL CORPUS LOADING...")
print("="*70)

try:
    # Import the actual class
    from spell_correction_system import CorpusService
    
    corpus_service = CorpusService()
    
    def progress_print(msg):
        print(f"   [PROGRESS] {msg}")
    
    corpus_text = corpus_service.load_corpus(progress_callback=progress_print)
    
    words_loaded = len(corpus_text.split())
    print(f"\n✅ SUCCESS: Loaded {words_loaded:,} words")
    
    if words_loaded >= 100000:
        print("✅ Meets minimum requirement (100,000 words)")
    else:
        print(f"⚠️  Below minimum requirement (needs 100,000, has {words_loaded:,})")
    
except Exception as e:
    print(f"\n❌ ERROR during loading: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)