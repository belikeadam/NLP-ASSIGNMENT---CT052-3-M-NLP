1. ADD STREAMLIT DEPLOYMENT FUNCTION (Insert after line ~2800, before def main())

# ============================================================================
# STREAMLIT WEB DEPLOYMENT
# ============================================================================

def create_streamlit_app():
    """Streamlit web deployment for spell checking"""
    try:
        import streamlit as st
        import plotly.graph_objects as go
    except ImportError:
        print("Error: Streamlit not installed. Run: pip install streamlit plotly")
        return
    
    st.set_page_config(
        page_title="Spelling Correction System",
        page_icon="✍️",
        layout="wide"
    )
    
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            padding: 1.5rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
            margin-bottom: 2rem;
        }
        .error-box {
            background: #fee;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #e74c3c;
            margin: 1rem 0;
        }
        .success-box {
            background: #efe;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #2ecc71;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header">Advanced Spelling Correction System</div>', unsafe_allow_html=True)
    
    @st.cache_resource
    def load_spell_checker():
        checker = AdvancedSpellChecker()
        if not checker.load_from_cache():
            corpus_service = CorpusService()
            corpus = corpus_service.load_corpus()
            checker.train(corpus)
        return checker
    
    with st.spinner("Loading spell checker..."):
        spell_checker = load_spell_checker()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter Text to Check")
        
        # Example buttons
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            if st.button("Example: Non-word Error"):
                st.session_state.input_text = "I recieved the grammer report seperate from the accomodation."
        with btn_col2:
            if st.button("Example: Real-word Error"):
                st.session_state.input_text = "I went too the store to buy there groceries."
        with btn_col3:
            if st.button("Clear"):
                st.session_state.input_text = ""
        
        user_input = st.text_area(
            "Your text (max 500 characters):",
            height=200,
            max_chars=500,
            key="input_text"
        )
        
        if st.button("Check Spelling", type="primary"):
            if user_input.strip():
                preprocessor = TextPreprocessor()
                words = re.findall(r'\b[a-zA-Z]+\b', user_input)
                
                errors_found = []
                corrected_text = user_input
                
                for word in words:
                    if not spell_checker.check_word(word):
                        suggestions = spell_checker.get_suggestions(word, user_input)
                        if suggestions:
                            errors_found.append({
                                'original': word,
                                'suggestions': suggestions[:3]
                            })
                            corrected_text = corrected_text.replace(word, f"**{suggestions[0].corrected}**", 1)
                
                if errors_found:
                    st.markdown(f'<div class="error-box">Found {len(errors_found)} spelling error(s)</div>', unsafe_allow_html=True)
                    
                    for error in errors_found:
                        st.write(f"**{error['original']}** →", 
                                ', '.join([f"{s.corrected} ({s.confidence:.2f})" for s in error['suggestions']]))
                    
                    st.markdown("### Corrected Text:")
                    st.markdown(corrected_text)
                else:
                    st.markdown('<div class="success-box">No spelling errors found!</div>', unsafe_allow_html=True)
    
    with col2:
        st.subheader("System Statistics")
        
        if spell_checker.is_trained:
            vocab_size = len(spell_checker.vocabulary)
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=vocab_size,
                title={'text': "Vocabulary Size"},
                gauge={
                    'axis': {'range': [None, 150000]},
                    'bar': {'color': "#667eea"},
                    'steps': [
                        {'range': [0, 50000], 'color': "#fee"},
                        {'range': [50000, 100000], 'color': "#ffe"},
                        {'range': [100000, 150000], 'color': "#dfd"}
                    ]
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Features")
            st.markdown("""
            - Edit Distance: Levenshtein & Damerau-Levenshtein
            - Context: Bigram & Trigram models
            - Real-word detection: to/too, their/there, etc.
            - Medical domain vocabulary
            """)
    
    st.markdown("---")
    st.markdown("**NLP Assignment - Part A, Question 1** | Built with Streamlit, NLTK & PyEnchant")
    
    2. MODIFY main() FUNCTION (Replace existing main() around line ~2900)

    def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(description='Advanced Spelling Correction System')
    parser.add_argument('--mode', type=str, default='gui', 
                       choices=['gui', 'train'],
                       help='Mode: gui (Tkinter) or train (train only)')
    parser.add_argument('--force-download', action='store_true')
    parser.add_argument('--skip-synthetic-detection', action='store_true')
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("Training mode - training spell checker...")
        corpus_service = CorpusService()
        corpus = corpus_service.load_corpus(
            force_download=args.force_download,
            download_if_synthetic=not args.skip_synthetic_detection
        )
        checker = AdvancedSpellChecker()
        checker.train(corpus)
        print("Training complete!")
    else:
        print("=" * 70)
        print("ADVANCED SPELLING CORRECTION SYSTEM")
        print("NLP Assignment - Part A, Question 1")
        print("=" * 70)
        print("\nLaunching Tkinter GUI...")
        app = SpellCheckerGUI(
            force_download=args.force_download,
            download_if_synthetic=not args.skip_synthetic_detection
        )
        app.run()

if __name__ == "__main__":
    # Check if running in Streamlit
    try:
        import streamlit as st
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is not None:
            create_streamlit_app()
        else:
            main()
    except ImportError:
        main()
    except Exception:
        main()

        3. UPDATE CONFIG CLASS (Modify existing Config class around line ~100)
CHANGE these values:

# BEFORE:
REALWORD_IMPROVEMENT_RATIO = 1.2
REALWORD_MIN_DELTA = 0.01

# AFTER:
REALWORD_IMPROVEMENT_RATIO = 1.15  # More sensitive
REALWORD_MIN_DELTA = 0.005  # More sensitive

4. UPDATE IMPORTS (Add at top with other imports around line ~30)
# ADD these imports if not present:
try:
    import streamlit as st
    import plotly.graph_objects as go
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

    # Tkinter GUI (existing):
python spelling_correction.py

# Streamlit Web App (NEW):
streamlit run spelling_correction.py

# Train only:
python spelling_correction.py --mode train