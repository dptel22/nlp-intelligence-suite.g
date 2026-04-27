import importlib.util
from pathlib import Path
import json

mod_path = Path(__file__).with_name('morphology.py')
spec = importlib.util.spec_from_file_location('morph', str(mod_path))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
Morph = m.MorphologyAnalyzer

cases = [
    ('empty', ''),
    ('short', 'Hello world'),
    ('normal', 'Natural language processing is fun and useful'),
    ('water', 'water'),
]

for name, text in cases:
    print('CASE:', name)
    ma = Morph(text)
    print('tokens =', ma._tokens)
    print('trigram_model =', ma._trigram_model)
    print('ngram_df n=1:\n', ma.get_ngram_df(n=1).to_string(index=False))
    print('ngram_df n=2:\n', ma.get_ngram_df(n=2).to_string(index=False))
    print('predict(language, processing):', ma.predict_next_word('Language', 'Processing'))
    print('predict(water,<none>):', ma.predict_next_word('water','<none>'))
    print('morpheme_analysis:\n', ma.get_morpheme_analysis().to_string(index=False))
    print('-'*60)

