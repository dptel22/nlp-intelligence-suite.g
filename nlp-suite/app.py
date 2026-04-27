import nltk, subprocess
for r in ['punkt','stopwords','averaged_perceptron_tagger','averaged_perceptron_tagger_eng','wordnet','movie_reviews','punkt_tab']:
    nltk.download(r, quiet=True)
subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], capture_output=True)

import streamlit as st
import plotly.express as px
import pandas as pd

st.title("NLP Intelligence Suite")
user_text = st.text_area("Paste your text here", height=200, key="user_input")

if st.button("Analyze", type="primary") and user_text.strip():
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Preprocessing","Morphology & LM","Syntactic Analysis","Text Generation","Corpus Stats"])

    with tab1:
        try:
            from pipeline.preprocessor import Preprocessor
            p = Preprocessor(user_text)
            tokens = p.tokenize()
            filtered = p.remove_stopwords(tokens)
            c1,c2,c3 = st.columns(3)
            c1.metric("Raw Tokens", len(tokens))
            c2.metric("After Stop Removal", len(filtered))
            c3.metric("Unique Words", len(set(tokens)))
            st.dataframe(p.get_comparison_df(), use_container_width=True)
        except Exception as e:
            st.error(f"Preprocessing error: {e}")
            st.exception(e)

    with tab2:
        try:
            from pipeline.morphology import MorphologyAnalyzer
            m = MorphologyAnalyzer(user_text)
            s1,s2,s3 = st.tabs(["Morphemes","N-Grams","Next Word"])
            with s1:
                st.dataframe(m.get_morpheme_analysis(), use_container_width=True)
            with s2:
                df_ng = m.get_ngram_df(n=2, top_k=12)
                fig = px.bar(df_ng, x="Count", y="NGram", orientation="h", title="Top Bigrams")
                fig.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True)
            with s3:
                with st.form("nw_form"):
                    w1 = st.text_input("Word 1")
                    w2 = st.text_input("Word 2")
                    if st.form_submit_button("Predict") and w1 and w2:
                        preds = m.predict_next_word(w1, w2)
                        st.table(pd.DataFrame(preds, columns=["Word", "Probability"]))
        except Exception as e:
            st.error(f"Morphology error: {e}")
            st.exception(e)

    with tab3:
        try:
            from pipeline.syntax import SyntaxAnalyzer
            s = SyntaxAnalyzer(user_text)
            chunks = s.get_chunks()
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Noun Phrases")
                for np_ in chunks["noun_phrases"]: st.write(f"• {np_}")
            with col2:
                st.subheader("Verb Phrases")
                for vp in chunks["verb_phrases"]: st.write(f"• {vp}")
            st.subheader("POS Tags")
            st.dataframe(s.get_pos_df(), use_container_width=True)
            st.subheader("Dependency Tree")
            dep_html = s.get_displacy_html()
            try:
                st.html(dep_html)
            except AttributeError:
                import streamlit.components.v1 as components
                components.html(dep_html, height=400, scrolling=True)
        except Exception as e:
            st.error(f"Syntax error: {e}")
            st.exception(e)

    with tab4:
        try:
            from pipeline.generator import TextGenerator
            wc = len(user_text.split())
            if wc >= 30:
                cache_key = f"gen_{hash(user_text)}"
                if cache_key not in st.session_state:
                    g = TextGenerator(user_text)
                    g.prepare_data()
                    g.build_model()
                    with st.spinner("Training LSTM (30 epochs)..."):
                        g.train(epochs=30)
                    st.session_state[cache_key] = g
                g = st.session_state[cache_key]
                length = st.slider("Generation length", 10, 50, 20)
                words = user_text.split()
                seed = (words[-2]+" "+words[-1]) if len(words)>=2 else words[-1]
                st.success(g.generate(seed, length))
            else:
                st.warning(f"Need 30+ words. You have {wc}.")
        except Exception as e:
            st.error(f"Text Generation error: {e}")
            st.exception(e)

    with tab5:
        try:
            from pipeline.corpus_manager import CorpusManager
            if "corpus" not in st.session_state:
                st.session_state.corpus = CorpusManager()
            corpus = st.session_state.corpus
            if st.button("Save to Corpus"):
                corpus.add_text(user_text)
                st.success("Saved!")
            if not corpus.is_empty():
                stats = corpus.get_stats()
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Texts", stats["total_texts"])
                c2.metric("Words", stats["total_words"])
                c3.metric("Vocab", stats["vocab_size"])
                c4.metric("TTR", stats["type_token_ratio"])
                fig2 = px.bar(corpus.get_top_unigrams(), x="Count", y="Word",
                              orientation="h", title="Top Words in Corpus")
                fig2.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig2, use_container_width=True)
                st.divider()
                st.subheader("Sentiment Classifier")
                if st.button("Train Sentiment Model"):
                    with st.spinner("Training on movie_reviews (~30s)..."):
                        r = corpus.train_sentiment_classifier()
                        st.session_state["sent_model"] = r["model"]
                        st.session_state["sent_vec"] = r["vectorizer"]
                        st.success(f"Accuracy: {r['accuracy']:.2%}")
                if "sent_model" in st.session_state:
                    pred = corpus.predict_sentiment(
                        user_text,
                        st.session_state["sent_model"],
                        st.session_state["sent_vec"]
                    )
                    st.metric("Sentiment", pred["label"].upper())
                    st.metric("Confidence", f"{pred['confidence']:.1%}")
            else:
                st.info("No texts in corpus yet. Click Save to Corpus above.")
        except Exception as e:
            st.error(f"Corpus error: {e}")
            st.exception(e)
