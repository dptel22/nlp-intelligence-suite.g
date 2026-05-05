from collections import Counter, defaultdict

import nltk
import pandas as pd
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

for r in ["punkt", "stopwords", "punkt_tab"]:
    nltk.download(r, quiet=True)


class MorphologyAnalyzer:
    def __init__(self, text: str):
        self.text = text
        self._tokens = self._get_content_tokens()
        self._trigram_model = self._build_trigram_model(self._tokens)
        self._follower_model = self._build_follower_model(self._tokens)

    def _get_content_tokens(self) -> list:
        stop = set(stopwords.words("english"))
        return [t.lower() for t in word_tokenize(self.text) if t.isalpha() and t.lower() not in stop]

    def _build_trigram_model(self, tokens: list) -> dict:
        if len(tokens) < 3:
            return {}

        model = defaultdict(lambda: defaultdict(float))
        for w1, w2, w3 in ngrams(tokens, 3):
            model[(w1, w2)][w3] += 1

        return self._normalize_model(model)

    def _build_follower_model(self, tokens: list) -> dict:
        if len(tokens) < 2:
            return {}

        model = defaultdict(lambda: defaultdict(float))
        for word, next_word in ngrams(tokens, 2):
            model[word][next_word] += 1

        return self._normalize_model(model)

    def _normalize_model(self, model: dict) -> dict:
        for key in model:
            total = sum(model[key].values())
            if total == 0:
                continue
            for word in list(model[key].keys()):
                model[key][word] /= total
        return dict(model)

    def get_ngram_df(self, n: int = 2, top_k: int = 15) -> pd.DataFrame:
        if len(self._tokens) < n:
            return pd.DataFrame(columns=["NGram", "Count"])
        counts = Counter([" ".join(g) for g in ngrams(self._tokens, n)])
        return pd.DataFrame(counts.most_common(top_k), columns=["NGram", "Count"])

    def predict_next_word(self, word1: str, word2: str, top_k: int = 3) -> list:
        key = (word1.strip().lower(), word2.strip().lower())
        preds = sorted(self._trigram_model.get(key, {}).items(), key=lambda x: x[1], reverse=True)
        if preds:
            return preds[:top_k]

        preds = sorted(self._follower_model.get(key[1], {}).items(), key=lambda x: x[1], reverse=True)
        if preds:
            return preds[:top_k]

        if not self._tokens:
            return [("Add more text first", 0.0)]
        return [("No matching context found", 0.0)]

    def _split_morphemes(self, word: str) -> dict:
        prefixes = ["under", "over", "inter", "super", "sub", "mis", "dis", "pre", "non", "out", "un"]
        suffixes = ["ation", "tion", "ness", "ment", "able", "ible", "less", "ful", "ing", "ers", "ian", "est", "ies", "er", "ed", "es", "ly", "s"]

        prefix = next((p for p in prefixes if word.startswith(p) and len(word) > len(p) + 3), "")
        suffix = next((s for s in suffixes if word.endswith(s) and len(word) > len(s) + 3 and not word.endswith("ss")), "")
        root = word

        if prefix:
            root = root[len(prefix):]
        if suffix:
            root = root[: -len(suffix)]
            if suffix == "ies":
                root += "y"

        if not prefix and not suffix:
            root = PorterStemmer().stem(word)
        if not root:
            root = PorterStemmer().stem(word)

        return {"root": root, "prefix": prefix or "-", "suffix": suffix or "-"}

    def get_morpheme_analysis(self, top_n: int = 10) -> pd.DataFrame:
        content = [t for t in self._tokens if len(t) > 3]
        top_words = [w for w, _ in Counter(content).most_common(top_n)]
        rows = []
        for word in top_words:
            parts = self._split_morphemes(word)
            rows.append({"Word": word, "Root": parts["root"], "Prefix": parts["prefix"], "Suffix": parts["suffix"]})
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
