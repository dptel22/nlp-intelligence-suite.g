import nltk, subprocess
for r in ['punkt','stopwords','averaged_perceptron_tagger','averaged_perceptron_tagger_eng','wordnet','movie_reviews','punkt_tab']:
    nltk.download(r, quiet=True)
subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], capture_output=True)

import streamlit as st
import plotly.express as px
import pandas as pd

# ── persistent state init ──────────────────────────────────────────────────
from pipeline.corpus_manager import CorpusManager
if "corpus" not in st.session_state:
    st.session_state.corpus = CorpusManager()
if "analysis" not in st.session_state:
    st.session_state.analysis = None   # holds all computed results
if "nw_result" not in st.session_state:
    st.session_state.nw_result = None  # next-word prediction result

corpus = st.session_state.corpus

st.title("NLP Intelligence Suite")
user_text = st.text_area("Paste your text here", height=200, key="user_input")

# ── Analyze button: runs pipeline and STORES results, never renders tabs itself ──
if st.button("Analyze", type="primary") and user_text.strip():
    with st.spinner("Running NLP pipeline..."):
        res = {"text": user_text}
        try:
            from pipeline.preprocessor import Preprocessor
            p = Preprocessor(user_text)
            tokens = p.tokenize()
            filtered = p.remove_stopwords(tokens)
            res["tokens"] = tokens
            res["filtered"] = filtered
            res["comparison_df"] = p.get_comparison_df()
        except Exception as e:
            res["pre_error"] = str(e)

        try:
            from pipeline.morphology import MorphologyAnalyzer
            m = MorphologyAnalyzer(user_text)
            res["morpheme_df"] = m.get_morpheme_analysis()
            res["ngram_df"] = m.get_ngram_df(n=2, top_k=12)
            res["morph_obj_key"] = f"morph_{hash(user_text)}"
            st.session_state[res["morph_obj_key"]] = m
        except Exception as e:
            res["morph_error"] = str(e)

        try:
            from pipeline.syntax import SyntaxAnalyzer
            s = SyntaxAnalyzer(user_text)
            res["chunks"] = s.get_chunks()
            res["pos_df"] = s.get_pos_df()
            res["dep_html"] = s.get_displacy_html()
        except Exception as e:
            res["syn_error"] = str(e)

        # LSTM: train only once per unique text
        gen_key = f"gen_{hash(user_text)}"
        if len(user_text.split()) >= 30 and gen_key not in st.session_state:
            try:
                from pipeline.generator import TextGenerator
                g = TextGenerator(user_text)
                g.prepare_data()
                g.build_model()
                g.train(epochs=30)
                st.session_state[gen_key] = g
            except Exception as e:
                res["gen_error"] = str(e)
        res["gen_key"] = gen_key

        st.session_state.analysis = res
        st.session_state.nw_result = None  # reset next-word on new analyze

# ── Render tabs whenever analysis exists in session ───────────────────────
if st.session_state.analysis and st.session_state.analysis["text"] == (user_text or st.session_state.analysis["text"]):
    res = st.session_state.analysis
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Preprocessing", "Morphology & LM", "Syntactic Analysis", "Text Generation", "Corpus Stats"])

    # ── TAB 1 ─────────────────────────────────────────────────────────────
    with tab1:
        if "pre_error" in res:
            st.error(res["pre_error"])
        else:
            c1,c2,c3 = st.columns(3)
            c1.metric("Raw Tokens", len(res["tokens"]))
            c2.metric("After Stop Removal", len(res["filtered"]))
            c3.metric("Unique Words", len(set(res["tokens"])))
            st.dataframe(res["comparison_df"], use_container_width=True)

    # ── TAB 2 ─────────────────────────────────────────────────────────────
    with tab2:
        if "morph_error" in res:
            st.error(res["morph_error"])
        else:
            s1, s2, s3 = st.tabs(["Morphemes", "N-Grams", "Next Word"])
            with s1:
                st.dataframe(res["morpheme_df"], use_container_width=True)
            with s2:
                fig = px.bar(res["ngram_df"], x="Count", y="NGram", orientation="h", title="Top Bigrams")
                fig.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True)
            with s3:
                # Use st.form so Predict doesn't collapse the tab
                with st.form("nw_form"):
                    w1 = st.text_input("Word 1", key="nw_w1")
                    w2 = st.text_input("Word 2", key="nw_w2")
                    submitted = st.form_submit_button("Predict")
                if submitted and w1 and w2:
                    m = st.session_state.get(res["morph_obj_key"])
                    if m:
                        preds = m.predict_next_word(w1, w2)
                        st.session_state.nw_result = preds
                if st.session_state.nw_result is not None:
                    st.table(pd.DataFrame(st.session_state.nw_result, columns=["Word", "Probability"]))

    # ── TAB 3 ─────────────────────────────────────────────────────────────
    with tab3:
        if "syn_error" in res:
            st.error(res["syn_error"])
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Noun Phrases")
                for np_ in res["chunks"]["noun_phrases"]: st.write(f"• {np_}")
            with col2:
                st.subheader("Verb Phrases")
                for vp in res["chunks"]["verb_phrases"]: st.write(f"• {vp}")
            st.subheader("POS Tags")
            st.dataframe(res["pos_df"], use_container_width=True)
            st.subheader("Dependency Tree")
            try:
                st.html(res["dep_html"])
            except AttributeError:
                import streamlit.components.v1 as components
                components.html(res["dep_html"], height=400, scrolling=True)

    # ── TAB 4 ─────────────────────────────────────────────────────────────
    with tab4:
        if "gen_error" in res:
            st.error(res["gen_error"])
        elif len(res["text"].split()) < 30:
            st.warning(f"Need 30+ words. You have {len(res['text'].split())}.")
        else:
            gen_key = res["gen_key"]
            if gen_key in st.session_state:
                g = st.session_state[gen_key]
                length = st.slider("Generation length", 10, 50, 20, key="gen_slider")
                words = res["text"].split()
                seed = (words[-2] + " " + words[-1]) if len(words) >= 2 else words[-1]
                st.success(g.generate(seed, length))
            else:
                st.info("Training failed or still in progress.")

    # ── TAB 5 ─────────────────────────────────────────────────────────────
    with tab5:
        if st.button("Save to Corpus", key="save_corpus_btn"):
            corpus.add_text(res["text"])
            st.success(f"Saved! Corpus now has {corpus.get_stats()['total_texts']} text(s).")

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
            if st.button("Train Sentiment Model", key="train_sent_btn"):
                with st.spinner("Training on movie_reviews (~30s)..."):
                    r = corpus.train_sentiment_classifier()
                    st.session_state["sent_model"] = r["model"]
                    st.session_state["sent_vec"] = r["vectorizer"]
                    st.success(f"Accuracy: {r['accuracy']:.2%}")
            if "sent_model" in st.session_state:
                pred = corpus.predict_sentiment(
                    res["text"],
                    st.session_state["sent_model"],
                    st.session_state["sent_vec"]
                )
                st.metric("Sentiment", pred["label"].upper())
                st.metric("Confidence", f"{pred['confidence']:.1%}")
        else:
            st.info("No texts in corpus yet. Click 'Save to Corpus' above.")
