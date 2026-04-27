import numpy as np


class TextGenerator:
    def __init__(self, text: str):
        self.text = text
        self.tokenizer = None
        self.model = None
        self.seq_length = 9
        self.vocab_size = 0
        self.X = None
        self.y = None

    def prepare_data(self):
        if len(self.text.split()) < 30:
            raise ValueError("Need at least 30 words for text generation.")
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.preprocessing.text import Tokenizer

        tok = Tokenizer()
        tok.fit_on_texts([self.text])
        self.tokenizer = tok
        self.vocab_size = len(tok.word_index) + 1
        seq = tok.texts_to_sequences([self.text])[0]
        sequences = []
        for i in range(self.seq_length, len(seq)):
            sequences.append(seq[i - self.seq_length : i + 1])
        sequences = np.array(
            pad_sequences(sequences, maxlen=self.seq_length + 1, padding="pre")
        )
        self.X = sequences[:, :-1]
        self.y = sequences[:, -1]

    def build_model(self):
        from tensorflow.keras.layers import Dense, Embedding, LSTM
        from tensorflow.keras.models import Sequential

        self.model = Sequential(
            [
                Embedding(self.vocab_size, 10, input_length=self.seq_length),
                LSTM(64),
                Dense(self.vocab_size, activation="softmax"),
            ]
        )
        self.model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )

    def train(self, epochs: int = 50):
        return self.model.fit(self.X, self.y, epochs=epochs, verbose=0)

    def generate(self, seed_text: str, num_words: int = 20) -> str:
        try:
            from tensorflow.keras.preprocessing.sequence import pad_sequences

            result = seed_text
            for _ in range(num_words):
                encoded = self.tokenizer.texts_to_sequences([result])[0]
                encoded = pad_sequences([encoded], maxlen=self.seq_length, padding="pre")
                probs = self.model.predict(encoded, verbose=0)
                idx = np.argmax(probs[0])
                word = self.tokenizer.index_word.get(idx, "")
                if word:
                    result += " " + word
            return result
        except Exception as e:
            return seed_text + f" [generation failed: {e}]"

    def is_trained(self) -> bool:
        return self.model is not None


if __name__ == "__main__":
    txt = " ".join(
        [
            "natural language processing is a field of artificial intelligence that helps computers understand human language"
        ]
        * 5
    )
    g = TextGenerator(txt)
    g.prepare_data()
    g.build_model()
    g.train(epochs=30)
    print(g.generate("natural language", 15))
