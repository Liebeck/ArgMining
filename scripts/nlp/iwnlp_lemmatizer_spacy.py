from iwnlp.iwnlp_wrapper import IWNLPWrapper


class IWNLPLemmatizerSpacy(object):
    @classmethod
    def load(cls, lemmatizer_path):
        lemmatizer = IWNLPWrapper(lemmatizer_path=lemmatizer_path)
        return cls(lemmatizer)

    def __init__(self, lemmatizer):
        self._lemmatizer = lemmatizer

    def __call__(self, doc):
        for token in doc:
            print(token.text.encode('utf-8'), token.pos_)
            lemmas = self._lemmatizer.lemmatize(token.text, pos_universal_google=token.pos_)
            print(lemmas)
            #token.iwnlp_lemma = lemmas

