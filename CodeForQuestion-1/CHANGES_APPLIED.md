# ============================================================================
# STREAMLIT WEB DEPLOYMENT - ENHANCED VERSION
# ============================================================================

def create_streamlit_app():
    """Enhanced Streamlit web deployment matching Tkinter GUI features"""
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
    
    # Custom CSS for better UI
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
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .error-word {
            background-color: #ffcccc;
            border-bottom: 2px solid #e74c3c;
            padding: 2px 4px;
            border-radius: 3px;
            font-weight: bold;
            cursor: pointer;
        }
        .success-box {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #2ecc71;
            margin: 1rem 0;
        }
        .error-box {
            background: linear-gradient(135deg, #fee 0%, #fdd 100%);
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #e74c3c;
            margin: 1rem 0;
        }
        .suggestion-box {
            background: #f8f9fa;
            padding: 0.5rem;
            border-radius: 5px;
            margin: 0.5rem 0;
            border-left: 3px solid #3498db;
        }
        .word-dict-item {
            padding: 0.3rem;
            border-bottom: 1px solid #eee;
            font-family: 'Courier New', monospace;
        }
        .char-counter {
            text-align: right;
            color: #666;
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header">Advanced Spelling Correction System<br><small style="font-size:0.5em;">NLP Assignment - Part A, Question 1</small></div>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'input_text' not in st.session_state:
        st.session_state.input_text = ""
    if 'last_checked_text' not in st.session_state:
        st.session_state.last_checked_text = ""
    if 'errors_cache' not in st.session_state:
        st.session_state.errors_cache = []
    if 'search_term' not in st.session_state:
        st.session_state.search_term = ""
    
    # Load spell checker (cached)
    @st.cache_resource
    def load_spell_checker():
        checker = AdvancedSpellChecker()
        if not checker.load_from_cache():
            with st.spinner("Training spell checker (first run only)..."):
                corpus_service = CorpusService()
                corpus = corpus_service.load_corpus()
                checker.train(corpus)
        return checker
    
    with st.spinner("Loading spell checker..."):
        spell_checker = load_spell_checker()
    
    # Main layout: 2 columns (text editor | sidebar)
    col_main, col_sidebar = st.columns([2, 1])
    
    # ========================================================================
    # LEFT PANEL: TEXT EDITOR
    # ========================================================================
    with col_main:
        st.subheader("Text Editor")
        
        # Example buttons row
        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
        
        with btn_col1:
            if st.button("📝 Non-word Example", use_container_width=True):
                st.session_state.input_text = "I recieved the grammer report seperate from the accomodation."
                st.rerun()
        
        with btn_col2:
            if st.button("🔄 Real-word Example", use_container_width=True):
                st.session_state.input_text = "I went too the store to buy there groceries."
                st.rerun()
        
        with btn_col3:
            if st.button("🏥 Medical Example", use_container_width=True):
                st.session_state.input_text = "The patiant has symtoms of diabetis and hypertention."
                st.rerun()
        
        with btn_col4:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.input_text = ""
                st.session_state.errors_cache = []
                st.rerun()
        
        # Text area with character counter
        user_input = st.text_area(
            "Your text (max 500 characters):",
            value=st.session_state.input_text,
            height=200,
            max_chars=500,
            key="text_input_area",
            help="Type or paste text. Spell checking happens automatically."
        )
        
        # Update session state
        st.session_state.input_text = user_input
        
        # Character counter
        char_count = len(user_input)
        st.markdown(f'<div class="char-counter">Characters: {char_count}/500</div>', unsafe_allow_html=True)
        
        # ========================================================================
        # AUTO SPELL CHECK (Real-time simulation)
        # ========================================================================
        if user_input.strip() and user_input != st.session_state.last_checked_text:
            st.session_state.last_checked_text = user_input
            
            # Analyze text
            preprocessor = TextPreprocessor()
            words = re.findall(r'\b[a-zA-Z]+\b', user_input)
            
            errors_found = []
            corrected_text = user_input
            
            for word in words:
                # Check for both non-word and real-word errors
                suggestions = spell_checker.get_suggestions(word, user_input)
                
                # Non-word error
                if not spell_checker.check_word(word):
                    if suggestions:
                        errors_found.append({
                            'word': word,
                            'type': 'non-word',
                            'suggestions': suggestions[:3]
                        })
                        # Highlight in text
                        corrected_text = corrected_text.replace(
                            word, 
                            f"**<span class='error-word'>{word}</span>**",
                            1
                        )
                
                # Real-word error (check for confusion pairs)
                elif suggestions and hasattr(suggestions[0], 'source') and suggestions[0].source == 'realword':
                    errors_found.append({
                        'word': word,
                        'type': 'real-word',
                        'suggestions': suggestions[:3]
                    })
                    corrected_text = corrected_text.replace(
                        word,
                        f"**<span class='error-word'>{word}</span>**",
                        1
                    )
            
            st.session_state.errors_cache = errors_found
        
        # ========================================================================
        # DISPLAY RESULTS
        # ========================================================================
        if st.session_state.errors_cache:
            st.markdown(f'<div class="error-box">⚠️ Found {len(st.session_state.errors_cache)} spelling error(s)</div>', unsafe_allow_html=True)
            
            # Auto-correct all button
            if st.button("✨ Auto-Correct All", type="primary", use_container_width=True):
                corrected_full = user_input
                for error in st.session_state.errors_cache:
                    if error['suggestions']:
                        corrected_full = re.sub(
                            r'\b' + error['word'] + r'\b',
                            error['suggestions'][0].corrected,
                            corrected_full,
                            count=1
                        )
                st.session_state.input_text = corrected_full
                st.session_state.errors_cache = []
                st.success(f"✅ Corrected {len(st.session_state.errors_cache)} word(s)!")
                st.rerun()
            
            # Show suggestions for each error
            st.markdown("### Detected Errors & Suggestions")
            
            for idx, error in enumerate(st.session_state.errors_cache):
                with st.expander(f"**{error['word']}** ({error['type']} error)", expanded=(idx < 3)):
                    if error['suggestions']:
                        st.markdown("**Top Suggestions:**")
                        for i, sug in enumerate(error['suggestions'][:3], 1):
                            confidence_color = "#2ecc71" if sug.confidence > 0.7 else "#f39c12" if sug.confidence > 0.5 else "#e74c3c"
                            st.markdown(f"""
                            <div class="suggestion-box">
                                {i}. <strong>{sug.corrected}</strong> 
                                <span style="color:{confidence_color}">● {sug.confidence:.2%}</span>
                                <br><small>Edit distance: {sug.edit_distance} | {sug.reason if hasattr(sug, 'reason') else 'suggestion'}</small>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Apply suggestion button
                            if st.button(f"Apply '{sug.corrected}'", key=f"apply_{idx}_{i}"):
                                st.session_state.input_text = re.sub(
                                    r'\b' + error['word'] + r'\b',
                                    sug.corrected,
                                    st.session_state.input_text,
                                    count=1
                                )
                                st.rerun()
                    else:
                        st.info("No suggestions available")
        
        elif user_input.strip():
            st.markdown('<div class="success-box">✅ No spelling errors found!</div>', unsafe_allow_html=True)
    
    # ========================================================================
    # RIGHT PANEL: SIDEBAR
    # ========================================================================
    with col_sidebar:
        # System statistics
        st.subheader("📊 System Statistics")
        
        if spell_checker.is_trained:
            vocab_size = len(spell_checker.vocabulary)
            
            # Vocabulary gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=vocab_size,
                title={'text': "Vocabulary Size", 'font': {'size': 14}},
                gauge={
                    'axis': {'range': [None, 150000]},
                    'bar': {'color': "#667eea"},
                    'steps': [
                        {'range': [0, 50000], 'color': "#fee"},
                        {'range': [50000, 100000], 'color': "#ffe"},
                        {'range': [100000, 150000], 'color': "#dfd"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 100000
                    }
                }
            ))
            fig.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
            
            # System features
            st.markdown("### ⚙️ Features")
            features = [
                ("Edit Distance", "Levenshtein & Damerau"),
                ("Context Model", "Bigram & Trigram"),
                ("Real-word Detection", "to/too, their/there"),
                ("Domain", "Medical vocabulary"),
                ("Corpus Size", f"{spell_checker.language_model.total_words:,} words")
            ]
            
            for feature, value in features:
                st.markdown(f"**{feature}:** {value}")
        
        st.markdown("---")
        
        # Word dictionary browser
        st.subheader("📚 Word Dictionary")
        
        # Search box
        search_term = st.text_input(
            "Search words:",
            value=st.session_state.search_term,
            placeholder="Type to search...",
            key="dict_search"
        )
        st.session_state.search_term = search_term
        
        # Get words
        all_words = spell_checker.get_all_words_sorted()
        
        # Filter words
        if search_term:
            filtered_words = [(w, f) for w, f in all_words if w.startswith(search_term.lower())][:100]
        else:
            filtered_words = all_words[:100]
        
        st.markdown(f"*Showing {len(filtered_words)} of {len(all_words)} words*")
        
        # Display words in scrollable container
        words_html = "<div style='max-height: 300px; overflow-y: scroll; border: 1px solid #ddd; padding: 10px; border-radius: 5px;'>"
        for word, freq in filtered_words:
            words_html += f"<div class='word-dict-item'>{word} <span style='color:#666;'>({freq})</span></div>"
        words_html += "</div>"
        
        st.markdown(words_html, unsafe_allow_html=True)
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #7f8c8d; padding: 1rem;'>
            <p style='margin:0; font-size:0.9rem;'>
                <strong>Natural Language Processing Assignment</strong> | Part A - Question 1
            </p>
            <p style='margin:0.5rem 0 0 0; font-size:0.8rem;'>
                Built with Streamlit, NLTK & PyEnchant | Real-time spell checking with bigram context
            </p>
        </div>
    """, unsafe_allow_html=True)