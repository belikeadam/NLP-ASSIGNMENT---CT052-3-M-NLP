# NLP Assignment Demonstration Script
## Part B - Individual Component (40 Marks)
### Duration: 10-15 Minutes Screen Recording

---

# 🎬 COMPLETE PRESENTATION SCRIPT

## OPENING (30 seconds)

> "Hello, my name is [YOUR NAME], and today I'll be demonstrating the Natural Language Processing systems developed for our assignment. This demonstration covers two main components: the Spelling Correction System worth 20 marks, and the Text Classification Model worth 20 marks.

> Both systems are deployed as web applications using Streamlit and are accessible online. Let me walk you through each system, explaining the techniques used, demonstrating the features, and discussing strengths, limitations, and potential improvements."

---

# PART 1: SPELLING CORRECTION SYSTEM (20 Marks)
## Estimated Time: 6-7 minutes

### 1.1 SYSTEM INTRODUCTION (45 seconds)

**[Navigate to: https://q1-spell-correction.streamlit.app/]**

> "This is our Spelling Correction System. As you can see from the header, the system is built using a single source of truth: the Kaggle Medical Transcriptions corpus.

> **Key Statistics visible:**
> - Vocabulary: approximately 21,000 unique words
> - Corpus Size: over 2.4 million words

> This system was built entirely from a medical science corpus, meeting the assignment requirement of using a domain-specific corpus with at least 100,000 words. We chose medical science because it offers specialized vocabulary and real-world application in healthcare NLP."

---

### 1.2 SYSTEM ARCHITECTURE & TECHNIQUES (90 seconds)

**[Point to System Capabilities section on the right side]**

> "Let me explain the NLP techniques implemented in this system:

> **1. Bigram Language Model with Laplace Smoothing:**
> - We use bigram probabilities to understand word context
> - Laplace smoothing handles unseen word combinations by adding a small count to prevent zero probabilities
> - This is crucial for real-word error detection

> **2. Damerau-Levenshtein Edit Distance:**
> - This is an enhanced edit distance algorithm that supports four operations: insertion, deletion, substitution, AND transposition
> - Traditional Levenshtein doesn't handle transposition efficiently
> - For example, 'hte' to 'the' is just ONE edit with Damerau-Levenshtein, not two

> **3. Three-Factor Scoring System:**
> - Edit Distance Score (40%): Prioritizes candidates closer to the misspelled word
> - Frequency Score (30%): Prefers common words from our corpus
> - Context Score (30%): Uses bigram probabilities to find contextually appropriate words

> **4. Real-Word Error Detection:**
> - We maintain 24 confusion pairs like 'their/there/they're', 'to/too/two'
> - System checks if an alternative word has significantly better bigram context (20% threshold)"

---

### 1.3 FEATURE DEMONSTRATION - NON-WORD ERRORS (90 seconds)

**[Click "📝 Non-word" example button]**

> "Let me demonstrate non-word error detection. I'll click on the 'Non-word' example button.

> The text says: 'I recieved the grammer report seperate from the accomodation.'

> Notice there are several misspelled words here. Let me click 'Check Spelling'."

**[Click "🔍 Check Spelling" button]**

> "The system has identified multiple spelling errors:

> 1. **'recieved'** - This is a common misspelling of 'received'. You can see:
>    - A confidence progress bar showing the overall score
>    - The scoring breakdown showing edit distance, frequency score, and context score
>    - The top suggestion is 'received' with high confidence based on the combined scoring

> 2. **'grammer'** should be 'grammar'
> 3. **'seperate'** should be 'separate' 
> 4. **'accomodation'** should be 'accommodation'

> I can click 'Apply' on any suggestion to correct individual words, or use 'Auto-Correct All' to fix everything at once."

**[Click "✨ Auto-Correct All"]**

> "All corrections have been applied. The system correctly identifies that these words don't exist in our corpus vocabulary and suggests the closest valid alternatives."

---

### 1.4 FEATURE DEMONSTRATION - REAL-WORD ERRORS (60 seconds)

**[Click "🗑️ Clear" then "🔄 Real-word" example button]**

> "Now let me demonstrate real-word error detection, which is more challenging because the misspelled words actually exist in the dictionary.

> The text says: 'The patient went too the clinic to see there doctor.'

> Here, 'too' should be 'to', and 'there' should be 'their'. These are real words used in the wrong context."

**[Click "🔍 Check Spelling"]**

> "The system uses bigram context analysis to detect these errors:
> - For 'too the', the bigram probability is much lower than 'to the' (5 vs 12,730 counts in corpus)
> - For 'there doctor', the bigram has zero occurrences, while 'their doctor' has 4

> The system requires a 50% improvement in bigram score to suggest a correction, preventing false positives on legitimate usage."

---

### 1.5 DICTIONARY SEARCH FEATURE (30 seconds)

**[Scroll to Dictionary Search section]**

> "As required by the assignment, the system provides a searchable dictionary of all corpus words with their frequencies.

**[Type 'patient' in the search box]**

> You can search for any word - for example, 'patient' appears over 22,000 times in our medical corpus. This confirms our medical domain specialization."

---

### 1.6 STRENGTHS & LIMITATIONS (60 seconds)

> "**Strengths of our Spelling Correction System:**

> 1. **Domain-Specific Accuracy**: Trained on real medical transcriptions, it recognizes medical terminology
> 2. **Context-Aware**: Uses bigram probabilities for intelligent suggestions
> 3. **Both Error Types**: Handles non-word AND real-word errors
> 4. **Fast Performance**: Corrections are generated in milliseconds
> 5. **User-Friendly GUI**: Clean interface with 500-character limit as required

> **Limitations:**

> 1. **Limited to Corpus Vocabulary**: Words not in the medical corpus may be flagged as errors
> 2. **Bigram Context Only**: Doesn't consider longer contexts (trigrams or beyond)
> 3. **No Part-of-Speech Awareness**: Cannot distinguish grammatical contexts
> 4. **Fixed Confusion Pairs**: Real-word detection limited to predefined pairs"

---

### 1.7 POTENTIAL IMPROVEMENTS - POS, IR, SEMANTICS (60 seconds)

> "**How could we improve this system using POS, IR, and Semantics?**

> **1. Part-of-Speech (POS) Tagging:**
> - Add a POS tagger to understand grammatical context
> - For example, after 'the', we expect a noun or adjective, not a verb
> - This would help detect errors like 'I went to there house' - knowing 'there' is an adverb and we need a possessive pronoun

> **2. Information Retrieval (IR) Techniques:**
> - Use TF-IDF weighting for word importance in suggestions
> - Implement document-level context for better understanding
> - Use inverted indices for faster candidate retrieval

> **3. Semantic Analysis:**
> - Integrate word embeddings like Word2Vec or GloVe
> - Words with similar meanings would be clustered in vector space
> - Could detect semantic misuse: 'The doctor prescribed medicine for his patience' - semantically, 'patience' doesn't fit medical context, should be 'patients'
> - Contextual embeddings like BERT could understand nuanced meanings"

---

# PART 2: TEXT CLASSIFICATION MODEL (20 Marks)
## Estimated Time: 6-7 minutes

### 2.1 SYSTEM INTRODUCTION (45 seconds)

**[Navigate to: https://q2-text-classification-system.streamlit.app/]**

> "Now let me demonstrate our Text Classification System for SMS Spam Detection.

> This system uses the UCI SMS Spam Collection dataset, a benchmark dataset containing 5,574 real SMS messages. The problem is binary classification: distinguishing spam messages from legitimate (ham) messages.

> As you can see in the header:
> - Current Model: Random Forest (our deployed model)
> - F1-Score: approximately 98.9%

> This is a supervised machine learning approach using five different algorithms. During training, we evaluated all five models and Random Forest achieved excellent performance with 98.9% F1-Score."

---

### 2.2 DATASET & EDA OVERVIEW (60 seconds)

**[Point to Model Statistics and Training Data sections]**

> "Let me explain our Exploratory Data Analysis findings:

> **Dataset Characteristics:**
> - Total: 5,572 messages
> - Ham (legitimate): 4,825 messages (86.6%)
> - Spam: 747 messages (13.4%)
> - This is an imbalanced dataset, which we handled appropriately

> **Key EDA Findings:**
> - Spam messages average 139 characters vs 71 for ham
> - Spam contains more promotional keywords ('FREE', 'WIN', 'PRIZE')
> - Spam uses more exclamation marks and capital letters
> - No missing values in the dataset

> The pie chart here shows the class distribution visually."

---

### 2.3 MODEL ARCHITECTURE & TECHNIQUES (90 seconds)

**[Point to Processing Pipeline section]**

> "Our preprocessing pipeline includes seven steps:

> 1. **Lowercase normalization** - Consistency in text
> 2. **URL & email removal** - But we keep markers indicating their presence
> 3. **Special character cleaning** - Remove noise
> 4. **Stopword removal** - Selective, keeping important words like 'free', 'win'
> 5. **Lemmatization** - Normalize word forms
> 6. **TF-IDF Vectorization** - Convert text to numerical features
> 7. **N-gram features** - Unigrams, bigrams, and trigrams

> **The Five Models We Built:**
> 1. Naive Bayes - Baseline probabilistic classifier
> 2. Logistic Regression - Linear model with regularization
> 3. Support Vector Machine - Maximum margin classifier
> 4. Random Forest - Ensemble of decision trees
> 5. Gradient Boosting - Sequential ensemble method

> Each model underwent hyperparameter tuning using Grid Search with 5-fold cross-validation."

---

### 2.4 SPAM DETECTION DEMONSTRATION (90 seconds)

**[Click "🚨 Spam Example" button]**

> "Let me demonstrate spam detection. I'll click on a spam example from our actual dataset."

**[Wait for text to appear, then click "🔍 Analyze Message"]**

> "The system classifies this as SPAM with high confidence. Look at the detection factors:
> - Contains promotional/monetary keywords
> - Uses urgent call-to-action phrases
> - May have excessive punctuation or capitalization

> The confidence meter shows our certainty level."

**[Click "✅ Ham Example" button, then "🔍 Analyze Message"]**

> "Now let's try a legitimate message. This is classified as HAM (legitimate):
> - Natural conversational language
> - Typical personal message structure  
> - No aggressive marketing language

> The model correctly distinguishes between these types with over 99% accuracy."

---

### 2.5 LITERATURE COMPARISON (60 seconds)

**[Click "📚 Literature" button to open the comparison modal]**

> "Our assignment required comparing results with previous works. Let me show you our literature comparison.

> **Key Benchmarks from Literature:**
> - Almeida et al. (2011): Naive Bayes achieved 96.5% accuracy
> - Cormack et al. (2007): SVM achieved 97.5% accuracy
> - Bhowmick & Hazarika (2016): Random Forest achieved 97.2%

> **Our Results:**
> - Random Forest (deployed): 98.9% accuracy, 98.9% F1-score
> - We trained 5 models total: SVM, Logistic Regression, Naive Bayes, Random Forest, Gradient Boosting
> - All our models exceed literature benchmarks

> This improvement is attributed to:
> 1. Enhanced preprocessing including spelling correction
> 2. Optimized TF-IDF parameters with n-grams up to trigrams
> 3. Comprehensive hyperparameter tuning using Grid Search
> 4. Feature engineering (URL markers, currency detection, etc.)"

**[Close the dialog]**

---

### 2.6 MODEL PERFORMANCE METRICS (45 seconds)

**[Point to the performance gauge and metrics]**

> "Looking at our deployed model's performance:

> - **Accuracy**: 98.9% - Overall correctness
> - **Precision**: 98.9% - Of messages flagged as spam, 98.9% were actually spam
> - **Recall**: 98.9% - We caught 98.9% of all spam messages
> - **ROC-AUC**: 0.993 - Excellent discrimination ability

> The confusion matrices in our report show:
> - Very few false positives (legitimate messages marked as spam)
> - Very few false negatives (spam messages missed)
> 
> Random Forest achieves this through ensemble learning - combining multiple decision trees to make robust predictions."

---

### 2.7 STRENGTHS & LIMITATIONS (60 seconds)

> "**Strengths of our Text Classification System:**

> 1. **High Accuracy**: 98.9% F1-score exceeds published benchmarks
> 2. **Multiple Models**: We compared 5 different algorithms
> 3. **Proper Tuning**: Grid search optimization for each model
> 4. **Real-time Deployment**: Instant classification on the web
> 5. **Interpretable**: Shows factors that influenced classification

> **Limitations:**

> 1. **Binary Classification Only**: Cannot handle multi-class scenarios
> 2. **English Only**: Not trained on other languages
> 3. **Static Model**: Doesn't adapt to new spam patterns
> 4. **Dataset Bias**: Trained on SMS format, may not generalize to email
> 5. **No Deep Learning**: Using traditional ML, not neural networks"

---

### 2.8 POTENTIAL IMPROVEMENTS (60 seconds)

> "**How could we improve this deployment further?**

> **1. Model Improvements:**
> - Implement deep learning models like BERT or LSTM
> - Use transfer learning from pre-trained language models
> - Add active learning to continuously improve from user feedback

> **2. Feature Engineering:**
> - Include metadata features (time sent, sender reputation)
> - Add character-level features for obfuscation detection
> - Implement word embeddings instead of TF-IDF

> **3. Deployment Enhancements:**
> - Add batch processing for multiple messages
> - Implement API endpoints for integration
> - Add model versioning and A/B testing
> - Include confidence calibration for better probability estimates

> **4. Real-world Considerations:**
> - Handle adversarial attacks (spammers trying to evade detection)
> - Implement periodic retraining on new data
> - Add explainability features (LIME, SHAP) for transparency"

---

# CLOSING (30 seconds)

> "In conclusion, I have demonstrated two NLP systems that meet all assignment requirements:

> **Spelling Correction System:**
> - Uses bigram language model and Damerau-Levenshtein edit distance
> - Handles both non-word and real-word errors
> - Built on a 2.4 million word medical corpus

> **Text Classification System:**
> - Achieves 98.9% accuracy on spam detection using Random Forest
> - Trained and compared 5 ML algorithms with hyperparameter tuning
> - Exceeds published literature benchmarks

> Both systems are deployed as web applications and demonstrate practical NLP applications. Thank you for watching this demonstration."

---

# 📋 QUICK REFERENCE CHECKLIST

## Before Recording:
- [ ] Test both URLs are accessible
- [ ] Clear browser cache
- [ ] Ensure stable internet connection
- [ ] Prepare examples mentioned in script
- [ ] Practice timing (aim for 12-13 minutes total)

## Part 1 Must Cover (20 marks):
- [ ] Show system capabilities
- [ ] Demonstrate non-word error detection
- [ ] Demonstrate real-word error detection  
- [ ] Show dictionary search feature
- [ ] Explain bigram model and edit distance
- [ ] Discuss strengths and limitations
- [ ] Explain POS/IR/Semantics improvements

## Part 2 Must Cover (20 marks):
- [ ] Explain dataset choice and EDA
- [ ] Show model performance metrics
- [ ] Demonstrate spam detection
- [ ] Demonstrate ham detection
- [ ] Show literature comparison
- [ ] Discuss 5 ML models used
- [ ] Discuss strengths and limitations
- [ ] Explain potential improvements

---

# 🎯 KEY TALKING POINTS BY MARKING CRITERIA

## Spelling Correction (20 marks):

**For "Distinction" (75-100):**
- Clearly explain benefits AND limitations of each technique
- Demonstrate ALL features clearly
- Show excellent knowledge of POS/IR/Semantics improvements
- Answer confidently if there are follow-up questions

**Must mention:**
1. Bigram language model with Laplace smoothing
2. Damerau-Levenshtein edit distance (including transposition)
3. Three-factor scoring: Edit Distance, Frequency, Context
4. Real-word detection using confusion pairs
5. Corpus-only approach (no external dictionaries)

## Text Classification (20 marks):

**For "Distinction" (75-100):**
- Clear understanding of deployment techniques
- Explain all features clearly
- Demonstrate model selection rationale

**Must mention:**
1. Five models trained: NB, LR, SVM, RF, GB
2. Grid Search hyperparameter tuning with 5-fold CV
3. TF-IDF with n-grams (unigrams, bigrams, trigrams)
4. Literature comparison with specific papers (Almeida, Cormack, Bhowmick)
5. Random Forest deployed as best performing model (~98.9% F1)
6. Why Random Forest works well: Ensemble of decision trees, handles non-linear patterns, resistant to overfitting

---

# ⏱️ TIMING GUIDE

| Section | Duration | Cumulative |
|---------|----------|------------|
| Opening | 0:30 | 0:30 |
| Q1 Introduction | 0:45 | 1:15 |
| Q1 Architecture | 1:30 | 2:45 |
| Q1 Non-word Demo | 1:30 | 4:15 |
| Q1 Real-word Demo | 1:00 | 5:15 |
| Q1 Dictionary | 0:30 | 5:45 |
| Q1 Strengths/Limits | 1:00 | 6:45 |
| Q1 Improvements | 1:00 | 7:45 |
| Q2 Introduction | 0:45 | 8:30 |
| Q2 Dataset/EDA | 1:00 | 9:30 |
| Q2 Architecture | 1:30 | 11:00 |
| Q2 Demo | 1:30 | 12:30 |
| Q2 Literature | 1:00 | 13:30 |
| Q2 Metrics | 0:45 | 14:15 |
| Q2 Strengths/Limits | 1:00 | 15:15 |
| Q2 Improvements | 1:00 | 16:15 |
| Closing | 0:30 | 16:45 |

**Target: 12-15 minutes** (can trim demos if running long)

---

# 🔑 TECHNICAL TERMS TO KNOW

If asked about any of these, be prepared to explain:

1. **Laplace Smoothing**: Adding 1 to all counts to avoid zero probabilities
2. **Damerau-Levenshtein**: Edit distance that includes transposition as single operation
3. **TF-IDF**: Term Frequency-Inverse Document Frequency - weights words by importance
4. **N-grams**: Contiguous sequences of n items (unigrams, bigrams, trigrams)
5. **Grid Search**: Exhaustive search over hyperparameter combinations
6. **Cross-Validation**: K-fold splitting to validate model performance
7. **F1-Score**: Harmonic mean of precision and recall
8. **ROC-AUC**: Area under Receiver Operating Characteristic curve
9. **Confusion Matrix**: Table showing TP, TN, FP, FN
10. **Regularization**: Technique to prevent overfitting (L1, L2)
