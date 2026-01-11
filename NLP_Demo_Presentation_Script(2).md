# NLP Assignment Demonstration Script
## Part B - Individual Component (40 Marks)
### Duration: 10-15 Minutes Screen Recording

**Presenter:** Mohamed Adam Bin Ajmal Khan  
**TP Number:** TP091722  
**Intake:** APDMP2508AI

---

# 🎬 COMPLETE PRESENTATION SCRIPT

## OPENING (30 seconds)

> "Hello, my name is Mohamed Adam, student ID TP091722 from intake APDMP2508AI. Today I'll be demonstrating two Natural Language Processing systems I developed for this assignment.

> The first is a Spelling Correction System, and the second is a Text Classification Model - each worth 20 marks. Both are deployed as web applications using Streamlit.

> I'll walk you through each system, show you how they work, and discuss their strengths, limitations, and potential improvements."

---

# PART 1: SPELLING CORRECTION SYSTEM (20 Marks)
## Estimated Time: 6-7 minutes

### 1.1 SYSTEM INTRODUCTION (45 seconds)

**[Navigate to: https://q1-spell-correction.streamlit.app/]**

> "Here's my Spelling Correction System. Up in the header, you can see it's built on a single source of truth - the Kaggle Medical Transcriptions corpus.

> Looking at the statistics: we have about 21,000 unique words in our vocabulary, trained on over 2.4 million words of text.

> I chose the medical domain because it has specialized vocabulary that's really useful for healthcare applications. This also meets the assignment requirement of using a domain-specific corpus with at least 100,000 words."

---

### 1.2 SYSTEM ARCHITECTURE & TECHNIQUES (90 seconds)

**[Point to System Capabilities section on the right side]**

> "Let me explain the key techniques. First, we have a **Bigram Language Model with Laplace Smoothing**. This helps us understand word context by looking at word pairs. Laplace smoothing prevents zero probabilities for unseen combinations.

> Second, we use **Damerau-Levenshtein Edit Distance**. Unlike regular Levenshtein, this handles transpositions as a single edit. So 'hte' to 'the' counts as just one edit, not two.

> Third, we have a **Three-Factor Scoring System**:
> - 40% for Edit Distance - closer matches score higher
> - 30% for Frequency - common words are preferred
> - 30% for Context - words that fit the surrounding text

> Finally, **Real-Word Error Detection** using 24 confusion pairs like 'their' versus 'there'. The system checks if an alternative has significantly better context."

---

### 1.3 FEATURE DEMONSTRATION - NON-WORD ERRORS (90 seconds)

**[Click "📝 Non-word" example button]**

> "Let me show you non-word error detection. I'll click the 'Non-word' example.

> This sentence has several misspelled words: recieved, grammer, seperate, and accomodation. Let me click 'Check Spelling'."

**[Click "🔍 Check Spelling" button]**

> "The system found multiple errors. Let's look at 'recieved' - a common misspelling of 'received'.

> You can see the confidence bar showing the overall score, and below that, a breakdown of the three scoring factors: edit distance, frequency, and context.

> The other errors are also detected: 'grammer' should be 'grammar', 'seperate' should be 'separate', and 'accomodation' should be 'accommodation'.

> I can apply corrections one by one, or use 'Auto-Correct All' to fix everything."

**[Click "✨ Auto-Correct All"]**

> "Done! All corrections applied. The system correctly identified words that don't exist in our vocabulary and suggested valid alternatives."

---

### 1.4 FEATURE DEMONSTRATION - REAL-WORD ERRORS (60 seconds)

**[Click "🗑️ Clear" then "🔄 Real-word" example button]**

> "Now let me demonstrate real-word error detection, which is more challenging because the misspelled words actually exist in the dictionary.

> The text says: 'The patient went too the clinic to see there doctor.'

> Here, 'too' should be 'to', and 'there' should be 'their'. These are real words - just used in the wrong context."

**[Click "🔍 Check Spelling"]**

> "The system detects these using bigram analysis. For 'too the', the bigram count is only 5 in our corpus, while 'to the' has over 12,000 occurrences. That's a huge difference.

> For 'there doctor', we have zero occurrences, but 'their doctor' has 4. The system requires at least 50% better context to suggest a correction - this prevents false positives."

---

### 1.5 DICTIONARY SEARCH FEATURE (30 seconds)

**[Scroll to Dictionary Search section]**

> "The assignment also requires a searchable dictionary. Here it is - you can look up any word and see how often it appears in our corpus.

**[Type 'patient' in the search box]**

> For example, 'patient' appears over 22,000 times. This confirms we're working with a medical domain."

---

### 1.6 STRENGTHS & LIMITATIONS (60 seconds)

> "Let me discuss the strengths. First, it's domain-specific - trained on real medical transcriptions. Second, it's context-aware using bigram probabilities. Third, it handles both non-word AND real-word errors. Fourth, it's fast - corrections in milliseconds. And fifth, the interface is clean with the required 500-character limit.

> For limitations: The vocabulary is limited to our corpus, so non-medical words might be flagged incorrectly. We only use bigrams, not longer contexts. There's no part-of-speech awareness. And real-word detection is limited to predefined confusion pairs."

---

### 1.7 POTENTIAL IMPROVEMENTS - POS, IR, SEMANTICS (60 seconds)

> "How could we improve this? Three main areas:

> **Part-of-Speech tagging** would help understand grammar. After 'the', we expect a noun - this would catch errors like 'I went to there house' because 'there' is an adverb, not a possessive.

> **Information Retrieval techniques** like TF-IDF could weight word importance better. We could use document-level context and inverted indices for faster lookups.

> **Semantic analysis** using word embeddings like Word2Vec would cluster similar words together. For example, it could catch 'The doctor prescribed medicine for his patience' - semantically, 'patience' doesn't fit a medical context, it should be 'patients'. BERT could understand even more nuanced meanings."

---

# PART 2: TEXT CLASSIFICATION MODEL (20 Marks)
## Estimated Time: 6-7 minutes

### 2.1 SYSTEM INTRODUCTION (45 seconds)

**[Navigate to: https://q2-text-classification-system.streamlit.app/]**

> "Now let's look at my Text Classification System for SMS Spam Detection.

> I'm using the UCI SMS Spam Collection - a benchmark dataset with about 5,500 real SMS messages. It's a binary classification problem: spam versus legitimate messages.

> Up in the header, you can see we're running Random Forest with a 98.9% F1-Score. I trained five different models and Random Forest performed the best."

---

### 2.2 DATASET & EDA OVERVIEW (60 seconds)

**[Point to Model Statistics and Training Data sections]**

> "About the dataset: We have about 5,500 messages total. 86% are legitimate - what we call 'ham' - and 14% are spam. So it's imbalanced, which we handled during training.

> From our EDA, spam messages are typically longer - averaging 139 characters versus 71 for ham. Spam uses more promotional words like 'FREE', 'WIN', 'PRIZE', and has more exclamation marks and capitals.

> The pie chart here shows this distribution visually."

---

### 2.3 MODEL ARCHITECTURE & TECHNIQUES (90 seconds)

**[Point to Processing Pipeline section]**

> "Our preprocessing has seven steps: lowercase normalization, URL and email removal - though we keep markers showing they were there - special character cleaning, selective stopword removal, lemmatization, TF-IDF vectorization, and n-gram features up to trigrams.

> We built five models: Naive Bayes as our baseline, Logistic Regression, SVM, Random Forest, and Gradient Boosting. Each one went through Grid Search hyperparameter tuning with 5-fold cross-validation."

---

### 2.4 SPAM DETECTION DEMONSTRATION (90 seconds)

**[Click "🚨 Spam Example" button]**

> "Let me show you spam detection. I'll click a spam example from our dataset."

**[Wait for text to appear, then click "🔍 Analyze Message"]**

> "Classified as SPAM with high confidence. You can see why - promotional keywords, urgent call-to-action, maybe excessive punctuation. The confidence meter shows how certain the model is."

**[Click "✅ Ham Example" button, then "🔍 Analyze Message"]**

> "Now a legitimate message. Classified as HAM. Natural language, normal message structure, no aggressive marketing. The model distinguishes these with over 99% accuracy."

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
> Our results: Random Forest at 98.9%. All five of our models actually exceed the literature benchmarks.

> We achieved this through enhanced preprocessing, optimized TF-IDF with trigrams, thorough hyperparameter tuning, and good feature engineering."

**[Close the dialog]**

---

### 2.6 MODEL PERFORMANCE METRICS (45 seconds)

**[Point to the performance gauge and metrics]**

> "Looking at the metrics: 98.9% accuracy, precision, and recall. ROC-AUC is 0.993 - excellent discrimination.

> Very few false positives - that means legitimate messages rarely get flagged as spam. And very few false negatives - spam rarely slips through.

> Random Forest works well here because it combines multiple decision trees, making predictions more robust."

---

### 2.7 STRENGTHS & LIMITATIONS (60 seconds)

> "Strengths: High accuracy at 98.9% F1-score, exceeding published benchmarks. We compared five different algorithms with proper Grid Search tuning. It's deployed for real-time classification and shows what influenced each prediction.

> Limitations: It's binary only - spam or not spam, no multi-class. English only. The model is static, doesn't adapt to new spam patterns. And it's trained on SMS, so might not generalize well to email. We're using traditional ML, not deep learning."

---

### 2.8 POTENTIAL IMPROVEMENTS (60 seconds)

> "For improvements: We could use deep learning like BERT or LSTM, or transfer learning from pre-trained models. Active learning would let the system improve from user feedback.

> For features: metadata like send time, character-level features for detecting obfuscation, and word embeddings instead of TF-IDF.

> For deployment: batch processing, API endpoints, model versioning, and A/B testing. Also, handling adversarial attacks, periodic retraining, and explainability with LIME or SHAP."

---

# CLOSING (30 seconds)

> "To wrap up: I've demonstrated two NLP systems that meet all the assignment requirements.

> My Spelling Correction System uses a bigram language model with Damerau-Levenshtein edit distance. It handles both non-word and real-word errors, built on a 2.4 million word medical corpus.

> My Text Classification System achieves 98.9% accuracy on spam detection using Random Forest. I trained and compared five ML algorithms with proper tuning, and all exceed published benchmarks.

> Both are deployed as web applications. Thank you for watching."

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
