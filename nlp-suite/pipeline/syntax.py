import pandas as pd


class SyntaxAnalyzer:
    def __init__(self, text: str):
        import spacy

        try:
            self._nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise OSError("spaCy model 'en_core_web_sm' not found. Fix: python -m spacy download en_core_web_sm")
        self.doc = self._nlp(text)

    def get_pos_df(self) -> pd.DataFrame:
        import spacy

        rows = [
            {
                "Token": t.text,
                "POS": t.pos_,
                "Tag": t.tag_,
                "Description": spacy.explain(t.tag_) or "",
            }
            for t in self.doc
            if not t.is_punct and not t.is_space
        ]
        return pd.DataFrame(rows)

    def get_chunks(self) -> dict:
        nps = list(dict.fromkeys([c.text for c in self.doc.noun_chunks]))
        vps = []
        for token in self.doc:
            if token.pos_ == "VERB":
                phrase = sorted(
                    [token] + [c for c in token.children if not c.is_punct],
                    key=lambda t: t.i,
                )
                vps.append(" ".join([t.text for t in phrase]))
        return {"noun_phrases": nps, "verb_phrases": list(dict.fromkeys(vps))}

    def get_displacy_html(self) -> str:
        from spacy import displacy

        return displacy.render(
            self.doc,
            style="dep",
            options={"compact": True, "distance": 120},
            page=False,
        )

    def get_ner_html(self) -> str:
        from spacy import displacy

        if not list(self.doc.ents):
            return "<p style='color:gray'>No named entities found.</p>"
        return displacy.render(self.doc, style="ent", page=False)

    def get_dependency_triples(self) -> list:
        return [
            {"token": t.text, "relation": t.dep_, "head": t.head.text}
            for t in self.doc
            if t.dep_ in ["nsubj", "dobj", "pobj", "attr"]
        ]


if __name__ == "__main__":
    s = SyntaxAnalyzer(
        "The intelligent student quickly solved the difficult programming problem."
    )
    print(s.get_pos_df())
    print(s.get_chunks())
