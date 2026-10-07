import re  # Regular expression - работа с текстом
import spacy  # Библиотека для работы с NLP - языковыми моделями
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from tqdm import tqdm


# Подготавливаем строки к модели
class TextDataProcessor(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "tok2vec"])
        self.stop_words = self.nlp.Defaults.stop_words

    def fit(self, X, y=None):
        return self

    def _basic_clean(self, text: str) -> str:

        text = text.lower()  # Делаем всё нижним регистром
        text = re.sub(r"[^a-z\s]", " ", text)  # удаляем из строк всё, кроме букв

        return text

    def _filter_doc(self, doc) -> str:

        clean_words = []

        for token in doc:
            lemma = token.lemma_  # Начальная форма слова

            if lemma and lemma not in self.stop_words and len(lemma.strip()) > 1:
                clean_words.append(lemma)

        return " ".join(clean_words)

    def transform(self, documents: pd.Series):

        texts = [self._basic_clean(doc) for doc in documents]

        results = []

        # Аналог батчей для текста - pipe. Берём 64 строки из общего DataStore
        for doc in tqdm(
            self.nlp.pipe(texts, batch_size=64, n_process=4),
            total=len(texts),
            desc="Cleaning text",
        ):
            results.append(self._filter_doc(doc))

        return results
