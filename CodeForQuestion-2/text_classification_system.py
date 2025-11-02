# ============================================================================
# TEXT CLASSIFICATION SYSTEM - COMPLETE IMPLEMENTATION
# Assignment: NLP - Part A, Question 2
# ============================================================================

"""
INSTALLATION INSTRUCTIONS:
1. Create a folder called 'text_classification'
2. Save this file as 'main.py' in that folder
3. Install required packages:
   pip install pandas numpy scikit-learn matplotlib seaborn plotly streamlit kagglehub nltk wordcloud imbalanced-learn

4. For training: python main.py --mode train
5. For deployment: streamlit run main.py --mode deploy

The system will automatically download the dataset from Kaggle on first run.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc, roc_auc_score)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# Text Processing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import string
from wordcloud import WordCloud

# Utilities
import os
import pickle
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import argparse
from collections import Counter

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    pass

# ============================================================================
# MODELS - Data Classes
# ============================================================================

@dataclass
class ModelMetrics:
    """Performance metrics for a model"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: List[List[int]]
    classification_report: str
    
    def to_dict(self):
        return asdict(self)

@dataclass
class ModelConfig:
    """Configuration for a model"""
    name: str
    model_class: Any
    param_grid: Dict
    description: str

# ============================================================================
# CORE - Interfaces (SOLID: Interface Segregation Principle)
# ============================================================================

class IPreprocessor(ABC):
    """Interface for text preprocessing"""
    
    @abstractmethod
    def transform(self, texts: List[str]) -> List[str]:
        """Transform raw texts to processed texts"""
        pass

class IModel(ABC):
    """Interface for classification models"""
    
    @abstractmethod
    def train(self, X, y) -> None:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Make predictions"""
        pass
    
    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        """Get prediction probabilities"""
        pass

class ITuningStrategy(ABC):
    """Interface for hyperparameter tuning strategies"""
    
    @abstractmethod
    def optimize(self, model, X, y, param_grid: Dict) -> Dict:
        """Optimize model hyperparameters"""
        pass

# ============================================================================
# SERVICES - Data Service (SOLID: Single Responsibility)
# ============================================================================

class KaggleDataService:
    """Handles dataset loading from Kaggle and local sources"""
    
    def __init__(self):
        self.cache_dir = "cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def load_spam_dataset(self) -> pd.DataFrame:
        """Load SMS Spam dataset"""
        print("📥 Loading Spam SMS Dataset...")
        
        cache_file = os.path.join(self.cache_dir, "spam_dataset.csv")
        
        if os.path.exists(cache_file):
            print("✅ Loading from cache...")
            df = pd.read_csv(cache_file)
        else:
            print("⬇️ Downloading from Kaggle...")
            try:
                import kagglehub
                # Download dataset from Kaggle
                path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")
                
                # Find the CSV file
                csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
                if csv_files:
                    df = pd.read_csv(os.path.join(path, csv_files[0]), encoding='latin-1')
                else:
                    raise FileNotFoundError("No CSV file found in downloaded dataset")
                
                # Standardize column names
                if 'v1' in df.columns and 'v2' in df.columns:
                    df = df.rename(columns={'v1': 'label', 'v2': 'text'})
                    df = df[['label', 'text']]
                
                # Save to cache
                df.to_csv(cache_file, index=False)
                print(f"💾 Dataset cached to {cache_file}")
                
            except Exception as e:
                print(f"⚠️ Could not download from Kaggle: {e}")
                print("📝 Creating sample dataset for demonstration...")
                df = self._create_sample_spam_dataset()
                df.to_csv(cache_file, index=False)
        
        print(f"✅ Dataset loaded: {len(df)} samples")
        return df
    
    def _create_sample_spam_dataset(self) -> pd.DataFrame:
        """Create a sample spam dataset for demonstration"""
        spam_samples = [
            "WINNER!! You have won a 1 million dollar prize! Call now!",
            "FREE entry to win £1000 cash prize! Text WIN to 12345",
            "Congratulations! You've been selected for a free iPhone. Click here!",
            "URGENT! Your account will be closed. Verify now at http://fake-site.com",
            "Hot singles in your area! Meet them tonight!",
            "Get rich quick! Invest now and earn thousands!",
            "SALE! 90% off everything! Limited time only!",
            "Your loan has been approved! Claim your money now!",
            "Free vacation to Bahamas! Just pay processing fee!",
            "Make $5000 working from home! No experience needed!"
        ] * 50
        
        ham_samples = [
            "Hey, are we still meeting for lunch tomorrow?",
            "Can you pick up some milk on your way home?",
            "Thanks for the birthday wishes! Had a great time!",
            "Meeting rescheduled to 3pm in conference room B",
            "I'll be there in 10 minutes",
            "Great presentation today! Well done!",
            "Don't forget to submit the report by Friday",
            "Happy to help! Let me know if you need anything",
            "See you at the gym this evening",
            "Dinner at 7? Let me know if that works"
        ] * 50
        
        labels = ['spam'] * len(spam_samples) + ['ham'] * len(ham_samples)
        texts = spam_samples + ham_samples
        
        df = pd.DataFrame({'label': labels, 'text': texts})
        return df.sample(frac=1, random_state=42).reset_index(drop=True)

# ============================================================================
# SERVICES - Preprocessing Service (SOLID: Single Responsibility)
# ============================================================================

class TextPreprocessingService(IPreprocessor):
    """Handles all text preprocessing operations"""
    
    def __init__(self, remove_stopwords: bool = True, use_lemmatization: bool = True):
        self.remove_stopwords = remove_stopwords
        self.use_lemmatization = use_lemmatization
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
    def transform(self, texts: List[str]) -> List[str]:
        """Apply all preprocessing steps"""
        return [self._preprocess_text(text) for text in texts]
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess a single text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]
        
        # Lemmatization
        if self.use_lemmatization:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        return ' '.join(tokens)

# ============================================================================
# UTILS - EDA Service (SOLID: Single Responsibility)
# ============================================================================

class AutomatedEDA:
    """Comprehensive Exploratory Data Analysis"""
    
    def __init__(self, output_dir: str = "eda_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_full_report(self, df: pd.DataFrame, text_col: str = 'text', 
                            label_col: str = 'label') -> Dict:
        """Generate comprehensive EDA report"""
        print("\n" + "="*70)
        print("📊 EXPLORATORY DATA ANALYSIS")
        print("="*70)
        
        report = {}
        
        # 1. Dataset Overview
        print("\n1️⃣ Dataset Overview")
        report['dataset_info'] = self._analyze_dataset_info(df)
        
        # 2. Class Distribution
        print("\n2️⃣ Class Distribution Analysis")
        report['class_distribution'] = self._analyze_class_distribution(df, label_col)
        
        # 3. Text Length Analysis
        print("\n3️⃣ Text Length Analysis")
        report['text_analysis'] = self._analyze_text_lengths(df, text_col, label_col)
        
        # 4. Word Frequency Analysis
        print("\n4️⃣ Word Frequency Analysis")
        report['word_freq'] = self._analyze_word_frequency(df, text_col, label_col)
        
        # 5. Missing Data Analysis
        print("\n5️⃣ Missing Data Analysis")
        report['missing_data'] = self._analyze_missing_data(df)
        
        # Generate visualizations
        print("\n6️⃣ Generating Visualizations...")
        self._generate_visualizations(df, text_col, label_col)
        
        # Save report
        report_file = os.path.join(self.output_dir, 'eda_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n✅ EDA Report saved to: {report_file}")
        print(f"📁 Visualizations saved to: {self.output_dir}/")
        
        return report
    
    def _analyze_dataset_info(self, df: pd.DataFrame) -> Dict:
        """Analyze basic dataset information"""
        info = {
            'num_samples': len(df),
            'num_features': len(df.columns),
            'columns': list(df.columns),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        }
        
        print(f"   📝 Total Samples: {info['num_samples']}")
        print(f"   📊 Features: {info['num_features']}")
        print(f"   💾 Memory Usage: {info['memory_usage']}")
        
        return info
    
    def _analyze_class_distribution(self, df: pd.DataFrame, label_col: str) -> Dict:
        """Analyze class distribution"""
        class_counts = df[label_col].value_counts()
        class_pcts = df[label_col].value_counts(normalize=True) * 100
        
        distribution = {
            'counts': class_counts.to_dict(),
            'percentages': class_pcts.to_dict(),
            'is_balanced': max(class_pcts) / min(class_pcts) < 1.5
        }
        
        print(f"\n   Class Distribution:")
        for cls, count in class_counts.items():
            pct = class_pcts[cls]
            print(f"   • {cls}: {count} ({pct:.1f}%)")
        
        if not distribution['is_balanced']:
            print(f"   ⚠️ Dataset is imbalanced!")
        else:
            print(f"   ✅ Dataset is balanced")
        
        return distribution
    
    def _analyze_text_lengths(self, df: pd.DataFrame, text_col: str, label_col: str) -> Dict:
        """Analyze text length statistics"""
        df['text_length'] = df[text_col].str.len()
        df['word_count'] = df[text_col].str.split().str.len()
        
        analysis = {
            'length_stats': df.groupby(label_col)['text_length'].describe().to_dict(),
            'word_count_stats': df.groupby(label_col)['word_count'].describe().to_dict()
        }
        
        print(f"\n   Text Length Statistics:")
        for cls in df[label_col].unique():
            cls_data = df[df[label_col] == cls]
            print(f"   • {cls}:")
            print(f"     - Avg length: {cls_data['text_length'].mean():.0f} chars")
            print(f"     - Avg words: {cls_data['word_count'].mean():.0f} words")
        
        return analysis
    
    def _analyze_word_frequency(self, df: pd.DataFrame, text_col: str, label_col: str) -> Dict:
        """Analyze word frequencies by class"""
        word_freq = {}
        
        for cls in df[label_col].unique():
            cls_texts = df[df[label_col] == cls][text_col]
            all_words = ' '.join(cls_texts).lower().split()
            word_freq[cls] = dict(Counter(all_words).most_common(20))
        
        print(f"\n   Top Words by Class:")
        for cls, words in word_freq.items():
            top_5 = list(words.items())[:5]
            print(f"   • {cls}: {', '.join([f'{w}({c})' for w, c in top_5])}")
        
        return word_freq
    
    def _analyze_missing_data(self, df: pd.DataFrame) -> Dict:
        """Analyze missing data"""
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        
        analysis = {
            'missing_counts': missing.to_dict(),
            'missing_percentages': missing_pct.to_dict(),
            'has_missing': missing.sum() > 0
        }
        
        if analysis['has_missing']:
            print(f"   ⚠️ Missing values found:")
            for col, count in missing[missing > 0].items():
                print(f"   • {col}: {count} ({missing_pct[col]:.1f}%)")
        else:
            print(f"   ✅ No missing values")
        
        return analysis
    
    def _generate_visualizations(self, df: pd.DataFrame, text_col: str, label_col: str):
        """Generate all visualizations"""
        # 1. Class Distribution Bar Chart
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Class distribution
        df[label_col].value_counts().plot(kind='bar', ax=axes[0, 0], color=['#3498db', '#e74c3c'])
        axes[0, 0].set_title('Class Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Class')
        axes[0, 0].set_ylabel('Count')
        
        # Text length distribution
        for cls in df[label_col].unique():
            cls_data = df[df[label_col] == cls]
            axes[0, 1].hist(cls_data[text_col].str.len(), alpha=0.6, label=cls, bins=50)
        axes[0, 1].set_title('Text Length Distribution by Class', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Text Length (characters)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        
        # Word count distribution
        df['word_count'] = df[text_col].str.split().str.len()
        for cls in df[label_col].unique():
            cls_data = df[df[label_col] == cls]
            axes[1, 0].hist(cls_data['word_count'], alpha=0.6, label=cls, bins=30)
        axes[1, 0].set_title('Word Count Distribution by Class', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Number of Words')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # Average metrics by class
        metrics = df.groupby(label_col).agg({
            'text_length': 'mean',
            'word_count': 'mean'
        }).round(0)
        
        x = np.arange(len(metrics.index))
        width = 0.35
        axes[1, 1].bar(x - width/2, metrics['text_length'], width, label='Avg Text Length', color='#3498db')
        axes[1, 1].bar(x + width/2, metrics['word_count'], width, label='Avg Word Count', color='#2ecc71')
        axes[1, 1].set_title('Average Metrics by Class', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Class')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(metrics.index)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'basic_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Word Clouds
        fig, axes = plt.subplots(1, len(df[label_col].unique()), figsize=(15, 6))
        if len(df[label_col].unique()) == 1:
            axes = [axes]
        
        for idx, cls in enumerate(df[label_col].unique()):
            cls_text = ' '.join(df[df[label_col] == cls][text_col])
            wordcloud = WordCloud(width=800, height=400, background_color='white',
                                colormap='viridis').generate(cls_text)
            axes[idx].imshow(wordcloud, interpolation='bilinear')
            axes[idx].set_title(f'Word Cloud - {cls}', fontsize=14, fontweight='bold')
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'wordclouds.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   ✅ Saved: basic_analysis.png")
        print("   ✅ Saved: wordclouds.png")

# ============================================================================
# SERVICES - Model Service (SOLID: Single Responsibility & Open/Closed)
# ============================================================================

class ModelTrainingService:
    """Handles model training and evaluation"""
    
    def __init__(self, vectorizer_type: str = 'tfidf'):
        self.vectorizer_type = vectorizer_type
        self.vectorizer = None
        self.models_config = self._get_models_config()
        self.trained_models = {}
        self.results = {}
        
    def _get_models_config(self) -> List[ModelConfig]:
        """Get configuration for all models"""
        return [
            ModelConfig(
                name='Naive Bayes',
                model_class=MultinomialNB,
                param_grid={'alpha': [0.1, 0.5, 1.0, 2.0]},
                description='Probabilistic classifier based on Bayes theorem'
            ),
            ModelConfig(
                name='Logistic Regression',
                model_class=LogisticRegression,
                param_grid={
                    'C': [0.1, 1, 10],
                    'solver': ['liblinear', 'saga'],
                    'max_iter': [1000]
                },
                description='Linear model for binary classification'
            ),
            ModelConfig(
                name='Support Vector Machine',
                model_class=SVC,
                param_grid={
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf'],
                    'probability': [True]
                },
                description='Maximum margin classifier'
            ),
            ModelConfig(
                name='Random Forest',
                model_class=RandomForestClassifier,
                param_grid={
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5]
                },
                description='Ensemble of decision trees'
            )
        ]
    
    def prepare_data(self, df: pd.DataFrame, text_col: str = 'text',
                    label_col: str = 'label', test_size: float = 0.2) -> Tuple:
        """Prepare data for training"""
        print("\n" + "="*70)
        print("🔧 DATA PREPARATION")
        print("="*70)
        
        # Preprocess texts
        print("\n1️⃣ Preprocessing texts...")
        preprocessor = TextPreprocessingService()
        df['processed_text'] = preprocessor.transform(df[text_col].tolist())
        
        # Split data
        print("2️⃣ Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], df[label_col],
            test_size=test_size, random_state=42, stratify=df[label_col]
        )
        
        print(f"   📊 Training samples: {len(X_train)}")
        print(f"   📊 Testing samples: {len(X_test)}")
        
        # Vectorize
        print(f"3️⃣ Vectorizing with {self.vectorizer_type.upper()}...")
        if self.vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
        else:
            self.vectorizer = CountVectorizer(max_features=3000, ngram_range=(1, 2))
        
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        print(f"   ✅ Feature dimensions: {X_train_vec.shape[1]}")
        
        return X_train_vec, X_test_vec, y_train, y_test
    
    def train_all_models(self, X_train, X_test, y_train, y_test) -> Dict:
        """Train and evaluate all models"""
        print("\n" + "="*70)
        print("🤖 MODEL TRAINING & EVALUATION")
        print("="*70)
        
        for config in self.models_config:
            print(f"\n{'='*70}")
            print(f"Training: {config.name}")
            print(f"Description: {config.description}")
            print(f"{'='*70}")
            
            # Create model
            model = config.model_class()
            
            # Train
            print("⏳ Training...")
            model.fit(X_train, y_train)
            
            # Evaluate
            print("📊 Evaluating...")
            metrics = self._evaluate_model(model, X_test, y_test, config.name)
            
            # Store
            self.trained_models[config.name] = model
            self.results[config.name] = {
                'config': config,
                'model': model,
                'metrics': metrics
            }
            
            # Print results
            self._print_metrics(config.name, metrics)
        
        return self.results
    
    def _evaluate_model(self, model, X_test, y_test, model_name: str) -> ModelMetrics:
        """Evaluate a single model"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = ModelMetrics(
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred, average='weighted'),
            recall=recall_score(y_test, y_pred, average='weighted'),
            f1_score=f1_score(y_test, y_pred, average='weighted'),
            roc_auc=roc_auc_score(y_test, y_pred_proba, average='weighted') if y_pred_proba is not None else 0.0,
            confusion_matrix=confusion_matrix(y_test, y_pred).tolist(),
            classification_report=classification_report(y_test, y_pred)
        )
        
        return metrics
    
    def _print_metrics(self, name: str, metrics: ModelMetrics):
        """Print model metrics"""
        print(f"\n📈 Results for {name}:")
        print(f"   • Accuracy:  {metrics.accuracy:.4f}")
        print(f"   • Precision: {metrics.precision:.4f}")
        print(f"   • Recall:    {metrics.recall:.4f}")
        print(f"   • F1-Score:  {metrics.f1_score:.4f}")
        print(f"   • ROC-AUC:   {metrics.roc_auc:.4f}")
    
    def compare_models(self) -> pd.DataFrame:
        """Compare all trained models"""
        comparison = []
        
        for name, result in self.results.items():
            metrics = result['metrics']
            comparison.append({
                'Model': name,
                'Accuracy': metrics.accuracy,
                'Precision': metrics.precision,
                'Recall': metrics.recall,
                'F1-Score': metrics.f1_score,
                'ROC-AUC': metrics.roc_auc
            })
        
        df_comparison = pd.DataFrame(comparison).round(4)
        return df_comparison.sort_values('F1-Score', ascending=False)
    
    def select_best_model(self) -> Tuple[str, Any, ModelMetrics]:
        """Select the best performing model"""
        best_model_name = None
        best_score = -1
        
        for name, result in self.results.items():
            # Weighted score: 30% Accuracy + 25% Precision + 25% Recall + 20% F1
            score = (0.3 * result['metrics'].accuracy +
                    0.25 * result['metrics'].precision +
                    0.25 * result['metrics'].recall +
                    0.2 * result['metrics'].f1_score)
            
            if score > best_score:
                best_score = score
                best_model_name = name
        
        best_result = self.results[best_model_name]
        return best_model_name, best_result['model'], best_result['metrics']

# ============================================================================
# SERVICES - Tuning Service (SOLID: Strategy Pattern)
# ============================================================================

class GridSearchTuning(ITuningStrategy):
    """Grid search hyperparameter tuning"""
    
    def optimize(self, model, X, y, param_grid: Dict) -> Dict:
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='f1_weighted',
            n_jobs=-1, verbose=0
        )
        grid_search.fit(X, y)
        return grid_search.best_params_

class RandomSearchTuning(ITuningStrategy):
    """Random search hyperparameter tuning"""
    
    def optimize(self, model, X, y, param_grid: Dict) -> Dict:
        random_search = RandomizedSearchCV(
            model, param_grid, n_iter=20, cv=5, scoring='f1_weighted',
            n_jobs=-1, random_state=42, verbose=0
        )
        random_search.fit(X, y)
        return random_search.best_params_

class HyperparameterTuningService:
    """Service for hyperparameter optimization"""
    
    def __init__(self, strategy: str = 'grid'):
        self.strategy = GridSearchTuning() if strategy == 'grid' else RandomSearchTuning()
    
    def tune_model(self, model_config: ModelConfig, X_train, y_train) -> Tuple[Any, Dict]:
        """Tune a model's hyperparameters"""
        print(f"\n🔍 Tuning {model_config.name}...")
        print(f"   Strategy: {self.strategy.__class__.__name__}")
        print(f"   Parameter grid: {model_config.param_grid}")
        
        model = model_config.model_class()
        best_params = self.strategy.optimize(model, X_train, y_train, model_config.param_grid)
        
        print(f"   ✅ Best parameters: {best_params}")
        
        # Train final model with best parameters
        final_model = model_config.model_class(**best_params)
        final_model.fit(X_train, y_train)
        
        return final_model, best_params
    
    def tune_all_models(self, models_config: List[ModelConfig], X_train, y_train) -> Dict:
        """Tune all models"""
        print("\n" + "="*70)
        print("⚙️ HYPERPARAMETER TUNING")
        print("="*70)
        
        tuned_models = {}
        
        for config in models_config:
            model, params = self.tune_model(config, X_train, y_train)
            tuned_models[config.name] = {
                'model': model,
                'params': params,
                'config': config
            }
        
        return tuned_models

# ============================================================================
# SERVICES - Visualization Service
# ============================================================================

class ModelVisualizationService:
    """Generate visualizations for model comparison"""
    
    def __init__(self, output_dir: str = "model_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def create_comparison_charts(self, comparison_df: pd.DataFrame):
        """Create model comparison visualizations"""
        print("\n📊 Generating comparison charts...")
        
        # 1. Metrics comparison bar chart
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
        
        for idx, (metric, color) in enumerate(zip(metrics, colors)):
            ax = axes[idx // 2, idx % 2]
            comparison_df.plot(x='Model', y=metric, kind='bar', ax=ax, color=color, legend=False)
            ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
            ax.set_xlabel('Model')
            ax.set_ylabel(metric)
            ax.set_ylim([0, 1.05])
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', padding=3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Radar chart for all metrics
        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        fig = go.Figure()
        
        for _, row in comparison_df.iterrows():
            values = [row[cat] for cat in categories]
            values.append(values[0])  # Close the polygon
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill='toself',
                name=row['Model']
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title='Model Performance Comparison (Radar Chart)',
            title_font_size=16
        )
        
        fig.write_html(os.path.join(self.output_dir, 'radar_comparison.html'))
        
        print(f"   ✅ Saved: model_comparison.png")
        print(f"   ✅ Saved: radar_comparison.html")
    
    def create_confusion_matrices(self, results: Dict):
        """Create confusion matrix visualizations"""
        n_models = len(results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (name, result) in enumerate(results.items()):
            cm = np.array(result['metrics'].confusion_matrix)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       cbar=True, square=True)
            axes[idx].set_title(f'{name}\nConfusion Matrix', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: confusion_matrices.png")

# ============================================================================
# SERVICES - Model Persistence Service
# ============================================================================

class ModelPersistenceService:
    """Save and load trained models"""
    
    def __init__(self, models_dir: str = "saved_models"):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
    
    def save_model(self, model: Any, vectorizer: Any, model_name: str, metrics: ModelMetrics):
        """Save model, vectorizer, and metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.models_dir, f"{model_name}_{timestamp}.pkl")
        
        model_package = {
            'model': model,
            'vectorizer': vectorizer,
            'model_name': model_name,
            'metrics': metrics.to_dict(),
            'timestamp': timestamp
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_package, f)
        
        print(f"💾 Model saved: {model_path}")
        return model_path
    
    def load_latest_model(self) -> Dict:
        """Load the most recent model"""
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.pkl')]
        
        if not model_files:
            raise FileNotFoundError("No saved models found")
        
        latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(self.models_dir, x)))
        model_path = os.path.join(self.models_dir, latest_model)
        
        with open(model_path, 'rb') as f:
            model_package = pickle.load(f)
        
        print(f"📂 Loaded model: {model_path}")
        return model_package

# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

class TrainingPipeline:
    """Complete training pipeline"""
    
    def __init__(self):
        self.data_service = KaggleDataService()
        self.eda_service = AutomatedEDA()
        self.model_service = ModelTrainingService()
        self.tuning_service = HyperparameterTuningService(strategy='grid')
        self.viz_service = ModelVisualizationService()
        self.persistence_service = ModelPersistenceService()
    
    def run_full_pipeline(self):
        """Execute complete training pipeline"""
        print("\n" + "="*70)
        print("🚀 TEXT CLASSIFICATION TRAINING PIPELINE")
        print("="*70)
        
        # Step 1: Load Data
        df = self.data_service.load_spam_dataset()
        
        # Step 2: Exploratory Data Analysis
        eda_report = self.eda_service.generate_full_report(df)
        
        # Step 3: Prepare Data
        X_train, X_test, y_train, y_test = self.model_service.prepare_data(df)
        
        # Step 4: Train Base Models
        results = self.model_service.train_all_models(X_train, X_test, y_train, y_test)
        
        # Step 5: Compare Models
        print("\n" + "="*70)
        print("📊 MODEL COMPARISON")
        print("="*70)
        comparison_df = self.model_service.compare_models()
        print("\n" + comparison_df.to_string(index=False))
        
        # Step 6: Hyperparameter Tuning
        tuned_results = self.tuning_service.tune_all_models(
            self.model_service.models_config, X_train, y_train
        )
        
        # Step 7: Evaluate Tuned Models
        print("\n" + "="*70)
        print("📈 TUNED MODEL EVALUATION")
        print("="*70)
        
        tuned_comparison = []
        for name, result in tuned_results.items():
            model = result['model']
            metrics = self.model_service._evaluate_model(model, X_test, y_test, name)
            
            tuned_comparison.append({
                'Model': name,
                'Accuracy': metrics.accuracy,
                'Precision': metrics.precision,
                'Recall': metrics.recall,
                'F1-Score': metrics.f1_score,
                'ROC-AUC': metrics.roc_auc
            })
            
            self.model_service._print_metrics(name + " (Tuned)", metrics)
            tuned_results[name]['metrics'] = metrics
        
        tuned_df = pd.DataFrame(tuned_comparison).round(4).sort_values('F1-Score', ascending=False)
        print("\n📊 Tuned Models Comparison:")
        print(tuned_df.to_string(index=False))
        
        # Step 8: Select Best Model
        print("\n" + "="*70)
        print("🏆 BEST MODEL SELECTION")
        print("="*70)
        
        best_model_name = tuned_df.iloc[0]['Model']
        best_model_result = tuned_results[best_model_name]
        
        print(f"\n✅ Best Model: {best_model_name}")
        print(f"   Parameters: {best_model_result['params']}")
        print(f"\n   Performance Metrics:")
        print(f"   • Accuracy:  {best_model_result['metrics'].accuracy:.4f}")
        print(f"   • Precision: {best_model_result['metrics'].precision:.4f}")
        print(f"   • Recall:    {best_model_result['metrics'].recall:.4f}")
        print(f"   • F1-Score:  {best_model_result['metrics'].f1_score:.4f}")
        print(f"   • ROC-AUC:   {best_model_result['metrics'].roc_auc:.4f}")
        
        # Step 9: Generate Visualizations
        self.viz_service.create_comparison_charts(tuned_df)
        self.viz_service.create_confusion_matrices(tuned_results)
        
        # Step 10: Save Best Model
        model_path = self.persistence_service.save_model(
            best_model_result['model'],
            self.model_service.vectorizer,
            best_model_name,
            best_model_result['metrics']
        )
        
        print("\n" + "="*70)
        print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\n📁 Results saved to:")
        print(f"   • EDA: eda_results/")
        print(f"   • Models: model_results/")
        print(f"   • Best Model: {model_path}")
        print(f"\n🚀 Ready for deployment! Run: streamlit run main.py --mode deploy")
        
        return best_model_result

# ============================================================================
# DEPLOYMENT - Streamlit Web Application
# ============================================================================

def create_deployment_app():
    """Create Streamlit web application for deployment"""
    try:
        import streamlit as st
    except ImportError:
        print("❌ Streamlit not installed. Install with: pip install streamlit")
        return
    
    # Page configuration
    st.set_page_config(
        page_title="Text Classification System",
        page_icon="🎯",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #2c3e50;
            text-align: center;
            padding: 1rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .prediction-box {
            padding: 2rem;
            border-radius: 10px;
            margin: 1rem 0;
            font-size: 1.2rem;
            color: #333;
        }
        .spam-box {
            background-color: #fee;
            border-left: 5px solid #e74c3c;
        }
        .ham-box {
            background-color: #efe;
            border-left: 5px solid #2ecc71;
        }
        .confidence-bar {
            height: 30px;
            border-radius: 15px;
            background-color: #ecf0f1;
            overflow: hidden;
            margin: 1rem 0;
        }
        .confidence-fill {
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            transition: width 0.5s ease;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">🎯 Text Classification System</div>', unsafe_allow_html=True)
    
    # Load model
    @st.cache_resource
    def load_model():
        try:
            persistence_service = ModelPersistenceService()
            return persistence_service.load_latest_model()
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            st.info("💡 Please run training first: python main.py --mode train")
            return None
    
    model_package = load_model()
    
    if model_package is None:
        return
    
    model = model_package['model']
    vectorizer = model_package['vectorizer']
    model_name = model_package['model_name']
    metrics = model_package['metrics']
    
    # Sidebar - Model Information
    with st.sidebar:
        st.header("📊 Model Information")
        st.write(f"**Model:** {model_name}")
        st.write(f"**Trained:** {model_package['timestamp']}")
        
        st.subheader("Performance Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            st.metric("Precision", f"{metrics['precision']:.3f}")
        with col2:
            st.metric("Recall", f"{metrics['recall']:.3f}")
            st.metric("F1-Score", f"{metrics['f1_score']:.3f}")
        
        if st.button("🔄 View Model Details"):
            st.session_state['show_details'] = not st.session_state.get('show_details', False)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Enter Text for Classification")
        
        # Example texts - dynamic selection
        import random
        
        spam_examples = [
            "WINNER!! You have won a $1000 prize! Call now to claim!",
            "FREE entry to win £1000 cash prize! Text WIN to 12345",
            "Congratulations! You've been selected for a free iPhone. Click here!",
            "URGENT! Your account will be closed. Verify now at http://fake-site.com",
            "Hot singles in your area! Meet them tonight!",
            "Get rich quick! Invest now and earn thousands!",
            "SALE! 90% off everything! Limited time only!",
            "Your loan has been approved! Claim your money now!",
            "Free vacation to Bahamas! Just pay processing fee!",
            "Make $5000 working from home! No experience needed!"
        ]
        
        ham_examples = [
            "Hey, are we still meeting for lunch tomorrow?",
            "Can you pick up some milk on your way home?",
            "Thanks for the birthday wishes! Had a great time!",
            "Meeting rescheduled to 3pm in conference room B",
            "I'll be there in 10 minutes",
            "Great presentation today! Well done!",
            "Don't forget to submit the report by Friday",
            "Happy to help! Let me know if you need anything",
            "See you at the gym this evening",
            "Dinner at 7? Let me know if that works"
        ]
        
        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            if st.button("💡 Try Spam Example"):
                st.session_state['input_text'] = random.choice(spam_examples)
        with col_ex2:
            if st.button("💡 Try Ham Example"):
                st.session_state['input_text'] = random.choice(ham_examples)
        
        # Text input
        user_input = st.text_area(
            "Your text:",
            value=st.session_state.get('input_text', ''),
            height=150,
            placeholder="Type or paste your text here..."
        )
        
        # Analyze button
        analyze_clicked = st.button("🔍 Analyze Text", type="primary", width='stretch')
        
        if analyze_clicked and user_input.strip():
            # Preprocess
            preprocessor = TextPreprocessingService()
            processed_text = preprocessor.transform([user_input])[0]
            
            # Vectorize
            text_vectorized = vectorizer.transform([processed_text])
            
            # Predict
            prediction = model.predict(text_vectorized)[0]
            
            # Get probability
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(text_vectorized)[0]
                confidence = max(probabilities) * 100
                spam_prob = probabilities[1] if len(probabilities) > 1 else (probabilities[0] if prediction == 'spam' else 1 - probabilities[0])
            else:
                confidence = 95.0
                spam_prob = 0.95 if prediction == 'spam' else 0.05
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Analysis Results")
            
            # Prediction box
            if prediction.lower() == 'spam':
                st.markdown(f"""
                    <div class="prediction-box spam-box">
                        <h2>🚨 SPAM DETECTED</h2>
                        <p>This message appears to be spam.</p>
                    </div>
                """, unsafe_allow_html=True)
                confidence_color = "#e74c3c"
            else:
                st.markdown(f"""
                    <div class="prediction-box ham-box">
                        <h2>✅ LEGITIMATE MESSAGE</h2>
                        <p>This message appears to be legitimate.</p>
                    </div>
                """, unsafe_allow_html=True)
                confidence_color = "#2ecc71"
            
            # Confidence meter
            st.markdown("### Confidence Level")
            st.markdown(f"""
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {confidence}%; background-color: {confidence_color};">
                        {confidence:.1f}%
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Explanation
            st.markdown("### 🔍 Key Factors")
            factors = []
            
            text_lower = user_input.lower()
            if any(word in text_lower for word in ['free', 'win', 'prize', 'winner', 'claim', 'cash']):
                factors.append("• Contains promotional/prize-related keywords")
            if any(word in text_lower for word in ['call now', 'click here', 'urgent', 'act now']):
                factors.append("• Uses urgent call-to-action phrases")
            if text_lower.count('!') > 2:
                factors.append("• Excessive use of exclamation marks")
            if len(user_input) < 20:
                factors.append("• Very short message length")
            if any(char.isdigit() for char in user_input):
                factors.append("• Contains numbers (common in promotions)")
            
            if not factors:
                factors.append("• Natural language patterns")
                factors.append("• Typical conversational structure")
            
            for factor in factors:
                st.markdown(factor)
            
            # Store in session
            st.session_state['last_prediction'] = {
                'text': user_input,
                'prediction': prediction,
                'confidence': confidence
            }
    
    with col2:
        st.subheader("📈 Quick Stats")
        
        # Model performance gauge
        import plotly.graph_objects as go
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=metrics['f1_score'] * 100,
            title={'text': "Model F1-Score"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#667eea"},
                'steps': [
                    {'range': [0, 50], 'color': "#fee"},
                    {'range': [50, 75], 'color': "#ffe"},
                    {'range': [75, 100], 'color': "#efe"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, width='stretch')
        
        # Classification distribution (example)
        st.markdown("### Sample Distribution")
        fig2 = go.Figure(data=[
            go.Pie(
                labels=['Ham', 'Spam'],
                values=[87, 13],
                hole=0.4,
                marker_colors=['#2ecc71', '#e74c3c']
            )
        ])
        fig2.update_layout(height=250, showlegend=True)
        st.plotly_chart(fig2, width='stretch')
    
    # Model details expander
    if st.session_state.get('show_details', False):
        st.markdown("---")
        st.subheader("🔬 Detailed Model Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Confusion Matrix**")
            cm = np.array(metrics['confusion_matrix'])
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, square=True)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
        
        with col2:
            st.markdown("**Classification Report**")
            st.text(metrics['classification_report'])
        
        with col3:
            st.markdown("**Feature Information**")
            st.write(f"Vectorizer: {vectorizer.__class__.__name__}")
            st.write(f"Features: {len(vectorizer.get_feature_names_out())}")
            st.write(f"Ngram Range: {vectorizer.ngram_range}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #7f8c8d;'>
            <p>🎓 NLP Text Classification System | Built with Streamlit & Scikit-learn</p>
        </div>
    """, unsafe_allow_html=True)

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Text Classification System')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'deploy'],
                       help='Mode: train or deploy')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("\n" + "="*70)
        print("🎓 TEXT CLASSIFICATION SYSTEM")
        print("NLP Assignment - Part A, Question 2")
        print("="*70)
        
        pipeline = TrainingPipeline()
        pipeline.run_full_pipeline()
        
    elif args.mode == 'deploy':
        print("\n🚀 Starting deployment server...")
        print("📱 Open your browser to view the app")
        create_deployment_app()

if __name__ == "__main__":
    # Check if running in Streamlit
    try:
        import streamlit as st
        # If we can access st.runtime, we're in Streamlit
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is not None:
            create_deployment_app()
        else:
            main()
    except:
        main()