import pandas as pd
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize

for r in ["punkt", "stopwords", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "wordnet", "punkt_tab"]:
    nltk.download(r, quiet=True)


class Preprocessor:
    def __init__(self, text: str):
        self.text = text

    def tokenize(self) -> list:
        if not self.text or not self.text.strip():
            return []
        return word_tokenize(self.text)

    def remove_stopwords(self, tokens: list) -> list:
        stop = set(stopwords.words("english"))
        return [t for t in tokens if t.isalpha() and t.lower() not in stop]

    def stem(self, tokens: list) -> list:
        ps = PorterStemmer()
        return [ps.stem(t) for t in tokens]

    def lemmatize(self, tokens: list) -> list:
        lem = WordNetLemmatizer()

        def get_pos(word):
            tag = nltk.pos_tag([word])[0][1][0].upper()
            return {
                "J": wordnet.ADJ,
                "V": wordnet.VERB,
                "N": wordnet.NOUN,
                "R": wordnet.ADV,
            }.get(tag, wordnet.NOUN)

        return [lem.lemmatize(t, get_pos(t)) for t in tokens]

    def get_comparison_df(self) -> pd.DataFrame:
        tokens = self.remove_stopwords(self.tokenize())
        if not tokens:
            return pd.DataFrame(columns=["Word", "Stem", "Lemma"])
        return pd.DataFrame(
            [{"Word": t, "Stem": self.stem([t])[0], "Lemma": self.lemmatize([t])[0]} for t in tokens]
        )

    def full_pipeline(self) -> dict:
        raw = self.tokenize()
        filtered = self.remove_stopwords(raw)
        return {
            "raw_tokens": raw,
            "filtered_tokens": filtered,
            "stems": self.stem(filtered),
            "lemmas": self.lemmatize(filtered),
            "comparison_df": self.get_comparison_df(),
        }


if __name__ == "__main__":
    p = Preprocessor("Running faster than expected, the studies showed much better results for everyone.")
    print(p.get_comparison_df().to_string())
