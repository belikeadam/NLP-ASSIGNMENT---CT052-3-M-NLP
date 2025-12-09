# How to Run - Advanced Spelling Correction System (Streamlit Web App)

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

Note: PyEnchant is recommended for best accuracy. Install it with:
```bash
pip install pyenchant
```

### 2. Run the Streamlit Web Application
```bash
streamlit run spell_correction_system.py
```

This will open the web application in your default browser at `http://localhost:8501`

### Optional: Training Mode Only
If you want to pre-train the model without opening the web interface:
```bash
python spell_correction_system.py --mode train
```

### Optional: Force real Kaggle download
If you want to force the download of the Kaggle dataset instead of using the local file or synthetic generator, use:
```bash
python spell_correction_system.py --force-download
```
Or if you want to prevent auto-detection of a synthetic corpus and skip the automatic download-on-detection, use:
```bash
python spell_correction_system.py --skip-synthetic-detection
```

## What to Expect

- **First Run**: The system will download/generate a medical corpus and train the language model (takes ~10-30 seconds)
- **Subsequent Runs**: Loads from cache instantly
- **Web Interface**: Opens a clean, professional Streamlit web application

## Features Available

- **Interactive web interface** with real-time spell checking
- **Click buttons to apply suggestions** for corrections
- **Auto-correct all** button for bulk corrections
- **Word dictionary browser** with search functionality
- **500 character limit** per session
- **Example buttons** for quick testing

## System Requirements

- Python 3.7+
- Streamlit
- NLTK library
- Plotly (for enhanced UI components)

## Troubleshooting

- If GUI doesn't open: Install tkinter (`sudo apt-get install python3-tk` on Linux)
- If corpus download fails: System automatically generates sample corpus
- For dependency issues: Run `pip install nltk kagglehub`

---

**NLP Assignment - Part A, Question 1**
