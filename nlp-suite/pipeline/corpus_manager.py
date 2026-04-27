import pandas as pd
from collections import Counter


class CorpusManager:
    def __init__(self):
        self.corpus = []
        self.all_words = []

    def add_text(self, text: str, label: str = "unlabeled"):
        import nltk
        from nltk.corpus import stopwords
        for r in ['punkt', 'stopwords', 'punkt_tab']:
            nltk.download(r, quiet=True)
        stop = set(stopwords.words('english'))
        tokens = [t.lower() for t in nltk.word_tokenize(text) if t.isalpha() and t.lower() not in stop]
        self.corpus.append({"text": text, "tokens": tokens, "label": label})
        self.all_words.extend(tokens)

    def get_stats(self) -> dict:
        tw = len(self.all_words)
        vs = len(set(self.all_words))
        tt = len(self.corpus)
        return {"total_texts": tt, "total_words": tw, "vocab_size": vs,
                "type_token_ratio": round(vs / tw, 4) if tw > 0 else 0.0,
                "avg_words_per_text": round(tw / tt, 1) if tt > 0 else 0.0}

    def get_top_unigrams(self, top_k: int = 10) -> pd.DataFrame:
        return pd.DataFrame(Counter(self.all_words).most_common(top_k), columns=["Word", "Count"])

    def is_empty(self) -> bool:
        return len(self.corpus) == 0

    def clear(self):
        self.corpus = []
        self.all_words = []

    def train_sentiment_classifier(self) -> dict:
        import nltk
        from nltk.corpus import movie_reviews
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        nltk.download('movie_reviews', quiet=True)
        docs, labels = [], []
        for cat in movie_reviews.categories():
            for fid in movie_reviews.fileids(cat):
                docs.append(" ".join(movie_reviews.words(fid)))
                labels.append(cat)
        vec = TfidfVectorizer(max_features=3000)
        X = vec.fit_transform(docs)
        Xtr, Xte, ytr, yte = train_test_split(X, labels, test_size=0.2, random_state=42)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Xtr, ytr)
        return {"accuracy": accuracy_score(yte, clf.predict(Xte)), "model": clf, "vectorizer": vec}

    def predict_sentiment(self, text: str, model, vectorizer) -> dict:
        X = vectorizer.transform([text])
        label_raw = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        return {"label": {"pos": "positive", "neg": "negative"}.get(label_raw, label_raw),
                "confidence": float(max(proba))}


if __name__ == "__main__":
    cm = CorpusManager()
    cm.add_text("I love natural language processing and machine learning.")
    cm.add_text("The weather today is terrible and gloomy outside.")
    print(cm.get_stats())
    r = cm.train_sentiment_classifier()
    print("Accuracy:", r["accuracy"])
    print(cm.predict_sentiment("This movie is absolutely amazing!", r["model"], r["vectorizer"]))
