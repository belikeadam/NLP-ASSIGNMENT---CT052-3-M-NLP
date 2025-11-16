# How to Run - Advanced Spelling Correction System

## Quick Start

### 1. Check Dependencies
```bash
python check_spell_dependencies.py
```

### 2. Run the Application
```bash
python spell_correction_system.py
```

## What to Expect

- **First Run**: The system will download/generate a medical corpus and train the language model (takes ~10-30 seconds)
- **Subsequent Runs**: Loads from cache instantly
- **GUI**: Opens a professional spelling correction interface

## Features Available

- **Real-time spell checking** with red underlines
- **Right-click misspelled words** for suggestions
- **Auto-correct all** button for bulk corrections
- **Word dictionary browser** with search functionality
- **500 character limit** per session

## System Requirements

- Python 3.7+
- NLTK library
- Tkinter (usually pre-installed with Python)

## Troubleshooting

- If GUI doesn't open: Install tkinter (`sudo apt-get install python3-tk` on Linux)
- If corpus download fails: System automatically generates sample corpus
- For dependency issues: Run `pip install nltk kagglehub`

---

**NLP Assignment - Part A, Question 1**
