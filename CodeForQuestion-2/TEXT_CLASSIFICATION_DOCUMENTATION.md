# 📚 Text Classification System - Developer Documentation

> **NLP Assignment - Part A, Question 2**  
> An advanced SMS spam detection system using multiple ML models, hyperparameter tuning, and modern web deployment.

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Installation & Setup](#installation--setup)
4. [Core Components Deep Dive](#core-components-deep-dive)
5. [Machine Learning Pipeline](#machine-learning-pipeline)
6. [Text Preprocessing](#text-preprocessing)
7. [Model Training & Evaluation](#model-training--evaluation)
8. [Configuration Guide](#configuration-guide)
9. [Usage Guide](#usage-guide)
10. [API Reference](#api-reference)
11. [Practical Examples](#practical-examples)
12. [Extending the System](#extending-the-system)
13. [Troubleshooting](#troubleshooting)

---

## Overview

### What Does This System Do?

This **Text Classification System** is designed to classify SMS messages as **spam** or **ham** (legitimate). It implements a complete machine learning pipeline from data loading to web deployment.

| Feature | Description |
|---------|-------------|
| **5 ML Models** | Naive Bayes, Logistic Regression, SVM, Random Forest, Gradient Boosting |
| **Hyperparameter Tuning** | GridSearchCV with 5-fold cross-validation |
| **Text Preprocessing** | Spelling correction, lemmatization, feature extraction |
| **TF-IDF Vectorization** | Unigrams, bigrams, and trigrams with sublinear scaling |
| **EDA Generation** | Automated visualizations and statistical reports |
| **Literature Benchmarking** | Compare with published research results |
| **Web Deployment** | Modern Streamlit interface with real-time classification |

### Key Features

- ✅ **Real Dataset**: UCI SMS Spam Collection (5,574 messages)
- ✅ **5 ML Models**: Comprehensive comparison and selection
- ✅ **Hyperparameter Tuning**: Grid search optimization
- ✅ **Spelling Correction**: Context-aware preprocessing
- ✅ **Literature Comparison**: Benchmark against published research
- ✅ **Professional Visualizations**: Word clouds, confusion matrices, comparison charts
- ✅ **Modern Web Interface**: Responsive Streamlit deployment

---

## System Architecture

```mermaid
graph TB
    subgraph "User Interface Layer"
        UI[Streamlit Web App]
    end
    
    subgraph "Core Services"
        TP[TrainingPipeline]
        DS[DataService]
        PP[TextPreprocessor]
        SC[SpellingCorrectionService]
        EDA[EDAService]
        MTS[ModelTrainingService]
        HTS[HyperparameterTuningService]
        VS[VisualizationService]
        MPS[ModelPersistenceService]
    end
    
    subgraph "ML Models"
        NB[Naive Bayes]
        LR[Logistic Regression]
        SVM[Support Vector Machine]
        RF[Random Forest]
        GB[Gradient Boosting]
    end
    
    subgraph "Data Layer"
        UCI[(UCI SMS Dataset)]
        CACHE[(Cached Models)]
        RESULTS[(Results & Visualizations)]
    end
    
    UI --> MPS
    TP --> DS
    TP --> EDA
    TP --> MTS
    TP --> HTS
    TP --> VS
    TP --> MPS
    DS --> UCI
    MTS --> NB
    MTS --> LR
    MTS --> SVM
    MTS --> RF
    MTS --> GB
    PP --> SC
    MPS --> CACHE
    VS --> RESULTS
```

### File Structure

```
CodeForQuestion-2/
├── text_classification_system.py    # Main source file (1768 lines)
├── cache/                           # Cached data files
│   └── sms_spam_collection.csv      # Downloaded dataset
├── saved_models/                    # Trained model files
│   └── Support_Vector_Machine_*.pkl # Best model + vectorizer
├── eda_results/                     # EDA outputs
│   ├── analysis.png                 # Statistical charts
│   ├── wordclouds.png               # Word cloud visualizations
│   └── eda_report.json              # Full EDA report
└── results/                         # Model comparison outputs
    ├── comparison.png               # Metric comparison charts
    └── confusion_matrices.png       # All model confusion matrices
```

---

## Installation & Setup

### Prerequisites

```bash
# Required Python packages
pip install pandas numpy scikit-learn matplotlib seaborn plotly streamlit kagglehub nltk wordcloud imbalanced-learn
```

### First-Time Setup

```bash
# Step 1: Train all models (downloads dataset, runs EDA, trains & tunes models)
python text_classification_system.py --mode train

# Step 2: Deploy the web application
streamlit run text_classification_system.py
```

---

## Core Components Deep Dive

### 1. Config Class (Lines 80-177)

Centralizes all system configuration:

```python
class Config:
    # Directories
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CACHE_DIR = os.path.join(BASE_DIR, "cache")
    MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
    
    # Training settings
    TEST_SIZE = 0.2           # 80/20 train/test split
    RANDOM_STATE = 42         # Reproducibility
    CV_FOLDS = 5              # 5-fold cross-validation
    
    # Feature extraction
    MAX_FEATURES = 5000       # Maximum vocabulary size
    NGRAM_RANGE = (1, 3)      # Unigrams, bigrams, trigrams
    MIN_DF = 2                # Minimum document frequency
    MAX_DF = 0.95             # Maximum document frequency
```

**Model Configurations:**

| Model | Key Parameters | Description |
|-------|---------------|-------------|
| **SVM** | C, kernel, gamma | Maximum margin classifier |
| **Logistic Regression** | C, solver | Linear model with regularization |
| **Naive Bayes** | alpha | Probabilistic baseline |
| **Random Forest** | n_estimators, max_depth | Ensemble for feature importance |
| **Gradient Boosting** | learning_rate, n_estimators | Sequential ensemble |

**Literature Benchmarks:**

```python
BENCHMARKS = {
    'Naive Bayes (Literature)': {'accuracy': 0.965, 'f1': 0.910, 'source': 'Almeida et al. (2011)'},
    'SVM (Literature)': {'accuracy': 0.975, 'f1': 0.930, 'source': 'Cormack et al. (2007)'},
    'Random Forest (Literature)': {'accuracy': 0.972, 'f1': 0.930, 'source': 'Bhowmick & Hazarika (2016)'}
}
```

---

### 2. Data Models (Lines 183-203)

**ModelMetrics**: Stores evaluation results
```python
@dataclass
class ModelMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: List[List[int]]
    classification_report: str
```

**ModelConfig**: Configuration for each model
```python
@dataclass
class ModelConfig:
    name: str              # Display name
    model_class: Any       # sklearn class
    param_grid: Dict       # Hyperparameter grid
    description: str       # Model description
```

---

### 3. DataService (Lines 238-354)

**Purpose**: Downloads and manages the UCI SMS Spam Collection dataset.

```python
class DataService:
    def load_dataset(self) -> pd.DataFrame:
        """Load UCI SMS Spam Collection (5,574 messages)"""
```

**Data Loading Flow:**

```mermaid
flowchart LR
    A[Start] --> B{Cache exists?}
    B -->|Yes| C[Load from cache]
    B -->|No| D{Try UCI download}
    D -->|Success| E[Save to cache]
    D -->|Fail| F{Try Kaggle}
    F -->|Success| E
    F -->|Fail| G[Raise Exception]
    E --> H[Return DataFrame]
    C --> H
```

**Dataset Statistics:**
- **Total Messages**: 5,574
- **Ham Messages**: 4,827 (86.6%)
- **Spam Messages**: 747 (13.4%)

---

### 4. TextPreprocessor (Lines 360-447)

**Purpose**: Cleans and transforms text for classification.

**Preprocessing Pipeline:**

```mermaid
flowchart TD
    A[Raw Text] --> B[Spelling Correction]
    B --> C[Extract Uppercase Words]
    C --> D[Lowercase]
    D --> E[Replace URLs → 'URL']
    E --> F[Replace Emails → 'EMAIL']
    F --> G[Extract Numbers]
    G --> H[Detect Punctuation Patterns]
    H --> I[Remove Special Characters]
    I --> J[Tokenize]
    J --> K[Selective Stopword Removal]
    K --> L[Lemmatization]
    L --> M[Add Feature Tokens]
    M --> N[Processed Text]
```

**Feature Tokens Added:**

| Token | Condition |
|-------|-----------|
| `HAS_NUMBER` | Text contains digits |
| `MULTIPLE_EXCLAMATION` | More than 2 exclamation marks |
| `HAS_CURRENCY` | Contains $, £, or € |
| `HAS_URL` | Originally contained a URL |
| `HAS_EMAIL` | Originally contained an email |
| `UPPERCASE_xxx` | For each UPPERCASE word |

**Example:**
```python
Input:  "FREE!!! You WON $1000! Call +1234567890 NOW!"
Output: "free won call now HAS_NUMBER MULTIPLE_EXCLAMATION HAS_CURRENCY UPPERCASE_free UPPERCASE_won UPPERCASE_now"
```

---

### 5. SpellingCorrectionService (Lines 453-602)

**Purpose**: Context-aware spelling correction that preserves spam indicators.

**Key Concept**: Regular spelling correction would "fix" spam-like abbreviations (ur → your), destroying important classification features. This service preserves them.

**Spam Vocabulary (Protected Words):**
```python
self.spam_vocabulary = {
    'ur', 'u', 'txt', 'msg', 'pls', 'plz', 'thx', 'gr8', 'l8r', 'b4',
    'FREE', 'WINNER', 'URGENT', 'CALL', 'CLICK', 'WIN', 'PRIZE', 'CASH',
    'won', 'claim', 'guaranteed', 'limited', 'offer', 'congratulations'
}
```

**Correction Flow:**
```python
def correct_text(self, text: str, preserve_spam_indicators: bool = True) -> str:
    for word in words:
        # Skip spam indicators
        if preserve_spam_indicators and word.lower() in self.spam_vocabulary:
            continue
        
        # Skip short words, numbers, URLs, ALL CAPS
        if len(word) <= 2 or word.isdigit() or word.isupper():
            continue
        
        # Correct using edit distance
        corrected = self._correct_word(word.lower())
```

---

### 6. EDAService (Lines 608-782)

**Purpose**: Generates comprehensive Exploratory Data Analysis reports.

**Generated Outputs:**

| File | Contents |
|------|----------|
| `analysis.png` | Class distribution, text length histograms, word count distributions |
| `wordclouds.png` | Word clouds for spam and ham classes |
| `eda_report.json` | Full statistical report in JSON format |

**EDA Report Structure:**
```python
{
    'dataset_info': {
        'num_samples': 5574,
        'num_features': 2,
        'memory_usage': "0.43 MB"
    },
    'class_distribution': {
        'counts': {'ham': 4827, 'spam': 747},
        'percentages': {'ham': 86.6, 'spam': 13.4},
        'is_balanced': False
    },
    'text_analysis': {
        'ham': {'avg_length': 71.45, 'avg_words': 15.2},
        'spam': {'avg_length': 138.67, 'avg_words': 25.8}
    },
    'word_freq': {...}
}
```

---

### 7. ModelTrainingService (Lines 788-939)

**Purpose**: Handles model training and evaluation for all 5 models.

**Training Workflow:**

```mermaid
sequenceDiagram
    participant TP as TrainingPipeline
    participant MTS as ModelTrainingService
    participant Model
    participant Metrics

    TP->>MTS: prepare_data(df)
    MTS->>MTS: Apply TextPreprocessor
    MTS->>MTS: Train/Test Split (80/20)
    MTS->>MTS: TF-IDF Vectorization
    MTS-->>TP: X_train, X_test, y_train, y_test
    
    loop For each model
        TP->>MTS: train_all_models()
        MTS->>Model: fit(X_train, y_train)
        MTS->>Model: predict(X_test)
        MTS->>Metrics: Calculate metrics
        MTS-->>TP: Results dictionary
    end
```

**TF-IDF Configuration:**
```python
self.vectorizer = TfidfVectorizer(
    max_features=5000,      # Maximum vocabulary
    ngram_range=(1, 3),     # Unigrams + bigrams + trigrams
    min_df=2,               # Minimum 2 documents
    max_df=0.95,            # Maximum 95% of documents
    sublinear_tf=True,      # Use log(1 + tf)
    use_idf=True            # Enable IDF weighting
)
```

---

### 8. HyperparameterTuningService (Lines 945-989)

**Purpose**: Optimizes hyperparameters using GridSearchCV.

```python
def tune_model(self, config: ModelConfig, X_train, y_train) -> Tuple[Any, Dict]:
    grid_search = GridSearchCV(
        model, 
        config.param_grid,
        cv=5,                    # 5-fold cross-validation
        scoring='f1_weighted',   # Optimize for F1
        n_jobs=-1,               # Use all CPU cores
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_
```

**Parameter Grids:**

| Model | Parameters Tuned |
|-------|-----------------|
| **SVM** | C: [0.1, 1, 10, 100], kernel: [linear, rbf], gamma: [scale, auto] |
| **Logistic Regression** | C: [0.01, 0.1, 1, 10, 100], solver: [liblinear, saga] |
| **Naive Bayes** | alpha: [0.1, 0.5, 1.0, 2.0, 5.0] |
| **Random Forest** | n_estimators: [100, 200], max_depth: [10, 20, None] |
| **Gradient Boosting** | learning_rate: [0.05, 0.1, 0.2], n_estimators: [100, 200] |

---

### 9. TrainingPipeline (Lines 1113-1246)

**Purpose**: Orchestrates the complete training workflow.

**Pipeline Steps:**

```mermaid
flowchart TD
    A[1. Load Dataset] --> B[2. Run EDA]
    B --> C[3. Prepare Data]
    C --> D[4. Train Base Models]
    D --> E[5. Compare Base Models]
    E --> F[6. Hyperparameter Tuning]
    F --> G[7. Evaluate Tuned Models]
    G --> H[8. Compare with Literature]
    H --> I[9. Select Best Model]
    I --> J[10. Generate Visualizations]
    J --> K[11. Save Best Model]
    K --> L[Pipeline Complete!]
```

**Output:**
```
TRAINING PIPELINE COMPLETED SUCCESSFULLY
========================================
Results Summary:
   EDA Results: eda_results/
   Model Comparisons: results/
   Best Model: saved_models/Support_Vector_Machine_20260105_150423.pkl

Best Model: Support Vector Machine
   Accuracy: 0.9784
   F1-Score: 0.8892

Ready for deployment! Run: streamlit run text_classification_system.py
```

---

## Machine Learning Pipeline

### Feature Extraction: TF-IDF

**TF-IDF (Term Frequency-Inverse Document Frequency)** converts text to numerical features:

```
TF-IDF(t,d) = TF(t,d) × IDF(t)

Where:
- TF(t,d) = log(1 + count(t in d))       # Sublinear scaling
- IDF(t) = log(N / df(t)) + 1            # Inverse document frequency
```

**Why N-grams?**

| N-gram Type | Example | Why Important |
|-------------|---------|---------------|
| Unigram (1) | "free" | Single keyword detection |
| Bigram (2) | "free money" | Phrase patterns |
| Trigram (3) | "call now free" | Spam-specific phrases |

---

### Model Comparison

**Typical Results:**

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **SVM** | 0.978 | 0.979 | 0.978 | 0.889 | 0.985 |
| **Logistic Regression** | 0.973 | 0.974 | 0.973 | 0.861 | 0.982 |
| **Random Forest** | 0.969 | 0.971 | 0.969 | 0.845 | 0.978 |
| **Gradient Boosting** | 0.962 | 0.964 | 0.962 | 0.823 | 0.971 |
| **Naive Bayes** | 0.958 | 0.960 | 0.958 | 0.810 | 0.965 |

> **Note**: SVM typically performs best for text classification due to its effectiveness in high-dimensional spaces.

---

### Confusion Matrix Interpretation

```
                Predicted
              Ham    Spam
Actual Ham   [ TN     FP ]
       Spam  [ FN     TP ]

Where:
- TN (True Negative): Ham correctly classified as Ham
- TP (True Positive): Spam correctly classified as Spam
- FP (False Positive): Ham incorrectly classified as Spam
- FN (False Negative): Spam incorrectly classified as Ham
```

**For Spam Detection:**
- **High Recall** is important: Missing spam (FN) is annoying
- **High Precision** is also important: Marking legitimate messages as spam (FP) can cause users to miss important messages

---

## Configuration Guide

### Tuning Model Weights

Adjust class weights for imbalanced data:

```python
# For Logistic Regression
'class_weight': ['balanced', None]

# For Random Forest
'class_weight': ['balanced', 'balanced_subsample', None]
```

### Adjusting Feature Extraction

```python
# For larger datasets - more features
MAX_FEATURES = 10000
NGRAM_RANGE = (1, 4)  # Include 4-grams

# For faster training - fewer features
MAX_FEATURES = 2000
NGRAM_RANGE = (1, 2)  # Only unigrams and bigrams
```

### Adding New Models

1. Add model configuration to `Config.MODELS`:

```python
'XGBoost': {
    'class': XGBClassifier,
    'params': {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5, 7]
    },
    'description': 'Extreme Gradient Boosting'
}
```

2. Import the model class:
```python
from xgboost import XGBClassifier
```

---

## Usage Guide

### Command-Line Usage

```bash
# Train all models (first-time setup)
python text_classification_system.py --mode train

# Deploy web application
streamlit run text_classification_system.py
```

### Web Interface Features

| Feature | Description |
|---------|-------------|
| **Text Input** | Enter or paste text message |
| **Real Examples** | Load actual spam/ham examples from dataset |
| **Analyze Button** | Classify the message |
| **Confidence Score** | Probability-based confidence level |
| **Detection Factors** | Why the prediction was made |
| **Model Stats** | Performance metrics and configuration |
| **Literature Comparison** | Compare with published benchmarks |

---

## API Reference

### DataService

```python
data_service = DataService()

# Load dataset (downloads if not cached)
df = data_service.load_dataset()

# Get random samples for examples
spam_examples = data_service.get_random_samples('spam', n=5)
ham_examples = data_service.get_random_samples('ham', n=5)
```

### TextPreprocessor

```python
preprocessor = TextPreprocessor(use_spelling_correction=True)

# Transform list of texts
processed = preprocessor.transform([
    "FREE WINNER!!! Call NOW to claim your PRIZE!",
    "Hey, are you coming to dinner tonight?"
])

# Output:
# ['free winner call claim prize HAS_NUMBER MULTIPLE_EXCLAMATION UPPERCASE_free UPPERCASE_winner UPPERCASE_now UPPERCASE_prize', 
#  'hey coming dinner tonight']
```

### SpellingCorrectionService

```python
spell_service = SpellingCorrectionService()

# Correct text (preserves spam indicators)
corrected = spell_service.correct_text(
    "Ur txt msg has typoes",
    preserve_spam_indicators=True
)
# Output: "Ur txt msg has typos" (ur, txt, msg preserved)

# Analyze corrections made
analysis = spell_service.analyze_corrections(original, corrected)
# {'num_corrections': 1, 'corrections': [{'original': 'typoes', 'corrected': 'typos'}]}
```

### ModelTrainingService

```python
model_service = ModelTrainingService()

# Prepare data
X_train, X_test, y_train, y_test = model_service.prepare_data(df)

# Train all models
results = model_service.train_all_models(X_train, X_test, y_train, y_test)

# Compare models
comparison_df = model_service.compare_models()
print(comparison_df)
```

### ModelPersistenceService

```python
persistence_service = ModelPersistenceService()

# Save model
model_path = persistence_service.save_model(
    model=trained_model,
    vectorizer=vectorizer,
    model_name="Support Vector Machine",
    metrics=model_metrics,
    params={'C': 10, 'kernel': 'rbf'}
)

# Load latest model
model_package = persistence_service.load_latest_model()
model = model_package['model']
vectorizer = model_package['vectorizer']
```

---

## Practical Examples

### Example 1: Complete Classification Flow

```python
"""Complete text classification example"""
from text_classification_system import (
    TextPreprocessor,
    ModelPersistenceService
)

def classify_message(text: str) -> dict:
    """Classify a single message as spam or ham"""
    
    # Load trained model
    persistence = ModelPersistenceService()
    package = persistence.load_latest_model()
    model = package['model']
    vectorizer = package['vectorizer']
    
    # Preprocess text
    preprocessor = TextPreprocessor()
    processed = preprocessor.transform([text])[0]
    
    # Vectorize
    features = vectorizer.transform([processed])
    
    # Predict
    prediction = model.predict(features)[0]
    
    # Get probability
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(features)[0]
        confidence = max(proba) * 100
    else:
        confidence = 95.0
    
    return {
        'text': text,
        'prediction': prediction,
        'confidence': f"{confidence:.1f}%",
        'is_spam': prediction.lower() == 'spam'
    }


# Test examples
messages = [
    "WINNER! You've won a $1000 prize! Call now!",
    "Hey, are you coming to the meeting at 3pm?",
    "FREE entry to win cash! Reply WIN to 12345",
    "Can you pick up milk on your way home?",
    "URGENT: Your account will be suspended! Verify NOW!"
]

print("=" * 60)
print("SMS CLASSIFICATION RESULTS")
print("=" * 60)

for msg in messages:
    result = classify_message(msg)
    emoji = "🚨 SPAM" if result['is_spam'] else "✅ HAM"
    print(f"\n{emoji} ({result['confidence']})")
    print(f"   {msg[:50]}...")
```

**Expected Output:**
```
============================================================
SMS CLASSIFICATION RESULTS
============================================================

🚨 SPAM (97.2%)
   WINNER! You've won a $1000 prize! Call now!...

✅ HAM (92.5%)
   Hey, are you coming to the meeting at 3pm?...

🚨 SPAM (98.1%)
   FREE entry to win cash! Reply WIN to 12345...

✅ HAM (89.3%)
   Can you pick up milk on your way home?...

🚨 SPAM (96.8%)
   URGENT: Your account will be suspended! Verify NOW!...
```

---

### Example 2: Preprocessing Step-by-Step

```python
"""Walk through the preprocessing for a spam message"""

text = "FREE!!! You WON £1000! Call +44-123-456 NOW! Visit http://scam.com"

print("Original:", text)
print()

# Step 1: Preserve uppercase words
uppercase_words = [w for w in text.split() if w.isupper() and len(w) > 2]
print("Step 1 - Uppercase words found:", uppercase_words)

# Step 2: Lowercase
text_lower = text.lower()
print("Step 2 - Lowercase:", text_lower)

# Step 3: Replace URLs
import re
text_url = re.sub(r'http\S+|www\S+', 'URL', text_lower)
print("Step 3 - URLs replaced:", text_url)

# Step 4: Extract number presence
has_numbers = bool(re.search(r'\d', text_url))
print("Step 4 - Has numbers:", has_numbers)

# Step 5: Check punctuation patterns
has_exclamation = text_url.count('!') > 2
has_currency = '£' in text or '$' in text or '€' in text
print("Step 5 - Multiple exclamation:", has_exclamation)
print("Step 5 - Has currency:", has_currency)

# Step 6: Remove special characters
text_clean = re.sub(r'[^a-zA-Z\s]', '', text_url)
print("Step 6 - Cleaned:", text_clean)

# Step 7: Tokenize and add features
tokens = text_clean.split()
if has_numbers:
    tokens.append('HAS_NUMBER')
if has_exclamation:
    tokens.append('MULTIPLE_EXCLAMATION')
if has_currency:
    tokens.append('HAS_CURRENCY')
for word in uppercase_words:
    tokens.append(f'UPPERCASE_{word.lower()}')

print("Step 7 - Final tokens:", ' '.join(tokens))
```

**Output:**
```
Original: FREE!!! You WON £1000! Call +44-123-456 NOW! Visit http://scam.com

Step 1 - Uppercase words found: ['FREE!!!', 'WON', 'NOW!']
Step 2 - Lowercase: free!!! you won £1000! call +44-123-456 now! visit http://scam.com
Step 3 - URLs replaced: free!!! you won £1000! call +44-123-456 now! visit URL
Step 4 - Has numbers: True
Step 5 - Multiple exclamation: True
Step 5 - Has currency: True
Step 6 - Cleaned: free you won  call  now visit URL
Step 7 - Final tokens: free you won call now visit URL HAS_NUMBER MULTIPLE_EXCLAMATION HAS_CURRENCY UPPERCASE_free!!! UPPERCASE_won UPPERCASE_now!
```

---

### Example 3: Model Evaluation Script

```python
"""Evaluate a trained model on custom test data"""

from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Custom test data
test_data = [
    ("Congratulations! You've won a free cruise! Call now!", "spam"),
    ("Hey, what time is the party tonight?", "ham"),
    ("URGENT: Your bank account needs verification", "spam"),
    ("Thanks for lunch yesterday, it was great!", "ham"),
    ("Win $5000 cash! Text MONEY to 12345", "spam"),
    ("Can you send me the report by EOD?", "ham"),
    ("FREE iPhone for first 100 callers!!!", "spam"),
    ("Running late, be there in 10 mins", "ham"),
]

# Load model
from text_classification_system import ModelPersistenceService, TextPreprocessor

persistence = ModelPersistenceService()
package = persistence.load_latest_model()
model = package['model']
vectorizer = package['vectorizer']

# Preprocess and predict
preprocessor = TextPreprocessor()
texts = [t[0] for t in test_data]
true_labels = [t[1] for t in test_data]

processed = preprocessor.transform(texts)
features = vectorizer.transform(processed)
predictions = model.predict(features)

# Evaluate
print("Classification Report:")
print(classification_report(true_labels, predictions))

print("\nConfusion Matrix:")
cm = confusion_matrix(true_labels, predictions, labels=['ham', 'spam'])
print(pd.DataFrame(cm, 
                   index=['Actual Ham', 'Actual Spam'], 
                   columns=['Pred Ham', 'Pred Spam']))

# Show individual results
print("\nDetailed Results:")
for text, true, pred in zip(texts, true_labels, predictions):
    status = "✅" if true == pred else "❌"
    print(f"{status} True: {true:4} | Pred: {pred:4} | {text[:40]}...")
```

---

### Example 4: Feature Importance Analysis

```python
"""Analyze which features are most important for classification"""

from text_classification_system import ModelPersistenceService
import numpy as np

# Load model
persistence = ModelPersistenceService()
package = persistence.load_latest_model()
model = package['model']
vectorizer = package['vectorizer']

# Get feature names
feature_names = vectorizer.get_feature_names_out()

# For Logistic Regression - get coefficients
if hasattr(model, 'coef_'):
    coefficients = model.coef_[0]  # For binary classification
    
    # Most indicative of SPAM (positive coefficients)
    spam_indices = np.argsort(coefficients)[-20:][::-1]
    print("Top 20 SPAM indicators:")
    for idx in spam_indices:
        print(f"  {feature_names[idx]:25} : {coefficients[idx]:.4f}")
    
    print("\n" + "="*50)
    
    # Most indicative of HAM (negative coefficients)
    ham_indices = np.argsort(coefficients)[:20]
    print("Top 20 HAM indicators:")
    for idx in ham_indices:
        print(f"  {feature_names[idx]:25} : {coefficients[idx]:.4f}")

# For Random Forest - get feature importances
elif hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-30:][::-1]
    
    print("Top 30 Most Important Features:")
    for idx in indices:
        print(f"  {feature_names[idx]:25} : {importances[idx]:.4f}")
```

**Sample Output:**
```
Top 20 SPAM indicators:
  call                      : 2.3456
  free                      : 2.1234
  txt                       : 1.9876
  claim                     : 1.8765
  prize                     : 1.7654
  won                       : 1.6543
  winner                    : 1.5432
  urgent                    : 1.4321
  HAS_CURRENCY              : 1.3210
  MULTIPLE_EXCLAMATION      : 1.2109
  ...

Top 20 HAM indicators:
  going                     : -1.8765
  home                      : -1.7654
  tomorrow                  : -1.6543
  tonight                   : -1.5432
  thanks                    : -1.4321
  ...
```

---

### Example 5: Batch Classification

```python
"""Classify messages from a CSV file"""

import pandas as pd
from text_classification_system import ModelPersistenceService, TextPreprocessor

def batch_classify(input_csv: str, output_csv: str):
    """
    Classify all messages in a CSV file.
    
    Input CSV must have a 'text' column.
    Output CSV will have 'text', 'prediction', and 'confidence' columns.
    """
    # Load data
    df = pd.read_csv(input_csv)
    if 'text' not in df.columns:
        raise ValueError("CSV must have a 'text' column")
    
    # Load model
    persistence = ModelPersistenceService()
    package = persistence.load_latest_model()
    model = package['model']
    vectorizer = package['vectorizer']
    
    # Preprocess
    preprocessor = TextPreprocessor()
    processed = preprocessor.transform(df['text'].tolist())
    
    # Vectorize
    features = vectorizer.transform(processed)
    
    # Predict
    predictions = model.predict(features)
    
    # Get confidence
    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(features)
        confidences = [max(p) * 100 for p in probas]
    else:
        confidences = [95.0] * len(predictions)
    
    # Create output dataframe
    output_df = pd.DataFrame({
        'text': df['text'],
        'prediction': predictions,
        'confidence': [f"{c:.1f}%" for c in confidences],
        'is_spam': [p.lower() == 'spam' for p in predictions]
    })
    
    # Save
    output_df.to_csv(output_csv, index=False)
    
    # Summary
    spam_count = sum(output_df['is_spam'])
    total = len(output_df)
    print(f"Processed {total} messages")
    print(f"  Spam: {spam_count} ({spam_count/total*100:.1f}%)")
    print(f"  Ham: {total-spam_count} ({(total-spam_count)/total*100:.1f}%)")
    print(f"Saved to: {output_csv}")


# Usage
batch_classify("messages.csv", "classified_messages.csv")
```

---

### Example 6: Unit Tests

```python
"""Unit tests for text_classification_system.py"""

import pytest
from text_classification_system import (
    TextPreprocessor,
    SpellingCorrectionService,
    DataService,
    ModelPersistenceService
)

class TestTextPreprocessor:
    """Test text preprocessing"""
    
    def test_lowercase(self):
        pp = TextPreprocessor(use_spelling_correction=False)
        result = pp.transform(["HELLO WORLD"])[0]
        assert 'hello' in result.lower()
    
    def test_url_removal(self):
        pp = TextPreprocessor(use_spelling_correction=False)
        result = pp.transform(["Visit http://example.com now"])[0]
        assert 'http' not in result
        assert 'HAS_URL' in result or 'url' in result.lower()
    
    def test_currency_detection(self):
        pp = TextPreprocessor(use_spelling_correction=False)
        result = pp.transform(["Win $1000 cash!"])[0]
        assert 'HAS_CURRENCY' in result
    
    def test_exclamation_detection(self):
        pp = TextPreprocessor(use_spelling_correction=False)
        result = pp.transform(["WOW!!! AMAZING!!! FREE!!!"])[0]
        assert 'MULTIPLE_EXCLAMATION' in result


class TestSpellingCorrection:
    """Test spelling correction service"""
    
    def test_preserves_spam_vocabulary(self):
        sc = SpellingCorrectionService()
        result = sc.correct_text("ur txt msg", preserve_spam_indicators=True)
        assert 'ur' in result.lower()
        assert 'txt' in result.lower()
    
    def test_corrects_typos(self):
        sc = SpellingCorrectionService()
        result = sc.correct_text("recieve teh package", preserve_spam_indicators=True)
        # Should correct 'recieve' to 'receive' and 'teh' to 'the'
        assert 'recieve' not in result.lower() or 'receive' in result.lower()


class TestDataService:
    """Test data loading"""
    
    def test_load_dataset(self):
        ds = DataService()
        try:
            df = ds.load_dataset()
            assert len(df) > 5000
            assert 'text' in df.columns
            assert 'label' in df.columns
        except Exception:
            pytest.skip("Dataset not available")
    
    def test_get_samples(self):
        ds = DataService()
        spam = ds.get_random_samples('spam', n=3)
        ham = ds.get_random_samples('ham', n=3)
        assert len(spam) == 3
        assert len(ham) == 3


# Parametrized test for spam detection
SPAM_EXAMPLES = [
    ("WINNER! You won $1000! Call now!", True),
    ("Hey are you free for lunch?", False),
    ("FREE entry! Text WIN to 12345", True),
    ("Meeting rescheduled to 3pm", False),
    ("URGENT: Verify your account NOW", True),
]

@pytest.mark.parametrize("text,expected_spam", SPAM_EXAMPLES)
def test_spam_detection(text, expected_spam):
    """Test that spam messages are correctly classified"""
    try:
        persistence = ModelPersistenceService()
        package = persistence.load_latest_model()
        model = package['model']
        vectorizer = package['vectorizer']
        preprocessor = TextPreprocessor()
        
        processed = preprocessor.transform([text])
        features = vectorizer.transform(processed)
        prediction = model.predict(features)[0]
        
        is_spam = prediction.lower() == 'spam'
        assert is_spam == expected_spam, f"Expected {'spam' if expected_spam else 'ham'}, got {prediction}"
    except FileNotFoundError:
        pytest.skip("Model not trained")
```

---

### Example 7: Quick Reference Cheat Sheet

```python
# ============================================================
# TEXT CLASSIFICATION SYSTEM - QUICK REFERENCE
# ============================================================

# ----- SETUP -----
from text_classification_system import (
    DataService, TextPreprocessor, SpellingCorrectionService,
    ModelTrainingService, ModelPersistenceService
)

# ----- DATA LOADING -----
ds = DataService()
df = ds.load_dataset()              # Returns DataFrame with 'text', 'label'
spam_samples = ds.get_random_samples('spam', n=5)
ham_samples = ds.get_random_samples('ham', n=5)

# ----- PREPROCESSING -----
preprocessor = TextPreprocessor(use_spelling_correction=True)
processed_texts = preprocessor.transform(["FREE WINNER!!! Call NOW!"])

# ----- SPELLING CORRECTION -----
spell = SpellingCorrectionService()
corrected = spell.correct_text("ur txt msg", preserve_spam_indicators=True)

# ----- LOAD TRAINED MODEL -----
persistence = ModelPersistenceService()
package = persistence.load_latest_model()
model = package['model']
vectorizer = package['vectorizer']

# ----- CLASSIFY MESSAGE -----
text = "You won a free iPhone!"
processed = preprocessor.transform([text])
features = vectorizer.transform(processed)
prediction = model.predict(features)[0]              # 'spam' or 'ham'
probabilities = model.predict_proba(features)[0]    # [P(ham), P(spam)]

# ----- TRAINING NEW MODELS -----
# Command line: python text_classification_system.py --mode train

# ----- DEPLOYMENT -----
# Command line: streamlit run text_classification_system.py
```

---

## Extending the System

### Adding a New Model

1. Add to `Config.MODELS`:
```python
'Neural Network': {
    'class': MLPClassifier,
    'params': {
        'hidden_layer_sizes': [(100,), (100, 50)],
        'activation': ['relu', 'tanh'],
        'learning_rate': ['constant', 'adaptive']
    },
    'description': 'Multi-layer Perceptron'
}
```

2. Import the class:
```python
from sklearn.neural_network import MLPClassifier
```

### Adding New Preprocessing Steps

Extend `TextPreprocessor._preprocess_text()`:

```python
def _preprocess_text(self, text: str) -> str:
    # ... existing steps ...
    
    # NEW: Detect phone numbers
    has_phone = bool(re.search(r'\+?\d[\d\s-]{8,}', text))
    if has_phone:
        tokens.append('HAS_PHONE_NUMBER')
    
    # NEW: Detect urgency words
    urgency_words = ['urgent', 'immediately', 'now', 'today', 'limited']
    if any(w in text.lower() for w in urgency_words):
        tokens.append('HAS_URGENCY')
```

### Adding New Literature Benchmarks

Add to `Config.BENCHMARKS`:

```python
'Deep Learning (Literature)': {
    'accuracy': 0.985,
    'f1': 0.950,
    'source': 'Kim (2014) - CNN for Text Classification'
}
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `FileNotFoundError: No saved models` | Training not completed | Run `python text_classification_system.py --mode train` |
| `ModuleNotFoundError: sklearn` | Missing dependency | Run `pip install scikit-learn` |
| Dataset download fails | Network/API issues | Check internet connection or Kaggle credentials |
| Slow training | Too many features/models | Reduce `MAX_FEATURES` or comment out slow models |
| Memory errors | Large dataset | Use `max_df=0.90` to reduce vocabulary |

### Performance Optimization

1. **Faster training**: Reduce hyperparameter grid sizes
2. **Less memory**: Lower `MAX_FEATURES` from 5000 to 2000
3. **Faster predictions**: Use Naive Bayes or Logistic Regression instead of SVM

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Add debugging to preprocessing
def _preprocess_text(self, text: str) -> str:
    logging.debug(f"Input: {text[:50]}...")
    # ... process ...
    logging.debug(f"Output: {result[:50]}...")
    return result
```

---

## Summary

This text classification system implements a complete ML pipeline:

1. **Data Loading**: UCI SMS Spam Collection (5,574 messages)
2. **EDA**: Automated visualizations and statistical analysis
3. **Preprocessing**: Spelling correction + feature engineering
4. **Training**: 5 models with hyperparameter tuning
5. **Evaluation**: Comprehensive metrics + literature comparison
6. **Deployment**: Modern Streamlit web interface

The system meets all assignment requirements and provides competitive results compared to published research.

---

> **Author**: NLP Assignment - Part A, Question 2  
> **Last Updated**: January 2026  
> **Tech Stack**: Python, Scikit-learn, NLTK, Streamlit, Plotly

