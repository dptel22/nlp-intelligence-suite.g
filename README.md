# NLP Intelligence Suite

A multi-tab Streamlit web app that runs a full NLP pipeline on any input text.

## Modules
- **Tab 1 - Preprocessing**: Tokenization, stopword removal, stemming vs lemmatization
- **Tab 2 - Morphology & LM**: N-gram analysis, trigram next-word prediction, morpheme breakdown
- **Tab 3 - Syntactic Analysis**: POS tagging, noun/verb phrase chunking, dependency tree
- **Tab 4 - Text Generation**: LSTM-based next-word generation
- **Tab 5 - Corpus Stats**: Vocabulary stats, TTR, sentiment classifier (trained on NLTK movie_reviews)

## Setup
pip install -r nlp-suite/requirements.txt
python -m spacy download en_core_web_sm
streamlit run nlp-suite/app.py

## Stack
Python · Streamlit · NLTK · spaCy · scikit-learn · TensorFlow · Plotly

