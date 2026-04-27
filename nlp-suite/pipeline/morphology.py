import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import ngrams, trigrams
from collections import Counter, defaultdict

for r in ["punkt", "stopwords", "punkt_tab"]:
    nltk.download(r, quiet=True)


class MorphologyAnalyzer:
    def __init__(self, text: str):
        self.text = text
        self._tokens = self._get_content_tokens()
        self._trigram_model = self._build_trigram_model()

    def _get_content_tokens(self) -> list:
        stop = set(stopwords.words("english"))
        return [t.lower() for t in word_tokenize(self.text) if t.isalpha() and t.lower() not in stop]

    def _build_trigram_model(self) -> dict:
        # If there are fewer than 3 tokens, building a trigram model isn't meaningful; return empty model
        if len(self._tokens) < 3:
            return {}

        model = defaultdict(lambda: defaultdict(float))
        for w1, w2, w3 in trigrams(self._tokens, pad_right=True, pad_left=True):
            model[(w1, w2)][w3] += 1

        for key in model:
            total = sum(model[key].values())
            if total == 0:
                continue
            for w in list(model[key].keys()):
                model[key][w] /= total

        return dict(model)

    def get_ngram_df(self, n: int = 2, top_k: int = 15) -> pd.DataFrame:
        if len(self._tokens) < n:
            return pd.DataFrame(columns=["NGram", "Count"])
        counts = Counter([" ".join(g) for g in ngrams(self._tokens, n)])
        return pd.DataFrame(counts.most_common(top_k), columns=["NGram", "Count"])

    def predict_next_word(self, word1: str, word2: str, top_k: int = 3) -> list:
        key = (word1.lower(), word2.lower())
        if key not in self._trigram_model:
            return [("No prediction available", 0.0)]
        preds = sorted(self._trigram_model[key].items(), key=lambda x: x[1], reverse=True)
        return preds[:top_k]

    def get_morpheme_analysis(self, top_n: int = 10) -> pd.DataFrame:
        ps = PorterStemmer()
        PREFIXES = ["under", "over", "mis", "dis", "pre", "re", "un", "out"]
        SUFFIXES = ["tion", "ness", "ment", "ful", "less", "ing", "est", "ly", "ed"]
        content = [t for t in self._tokens if len(t) > 3]
        top_words = [w for w, _ in Counter(content).most_common(top_n)]
        rows = []
        for word in top_words:
            prefix = next((p for p in PREFIXES if word.startswith(p) and len(word) > len(p) + 2), "")
            suffix = next((s for s in SUFFIXES if word.endswith(s) and len(word) > len(s) + 3), "")
            rows.append({"Word": word, "Root": ps.stem(word), "Prefix": prefix, "Suffix": suffix})
        return pd.DataFrame(rows)


if __name__ == "__main__":
    text = (
        "Natural language processing is a fascinating field of artificial intelligence. "
        "Researchers studied language models extensively. Processing requires careful "
        "preprocessing and analysis of linguistic patterns repeatedly."
    )
    m = MorphologyAnalyzer(text)
    print(m.get_ngram_df(n=2))
    print(m.predict_next_word("language", "processing"))
    print(m.get_morpheme_analysis())
