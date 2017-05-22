class Token:
    def __init__(self, token_index_in_sentence, text, pos_tag=None, mate_tools_pos_tag=None, mate_tools_lemma=None,
                 tree_tagger_lemma=None,
                 iwnlp_lemma=None, polarity=None, spacy_pos_stts=None, spacy_pos_universal_google=None,
                 spacy_ner_type=None,
                 spacy_ner_iob=None, spacy_shape=None, spacy_is_punct=None, spacy_like_url=None, spacy_like_num=None):
        self.token_index_in_sentence = token_index_in_sentence
        self.text = text
        self.pos_tag = pos_tag
        self.mate_tools_pos_tag = mate_tools_pos_tag
        self.mate_tools_lemma = mate_tools_lemma
        self.tree_tagger_lemma = tree_tagger_lemma
        self.iwnlp_lemma = iwnlp_lemma
        self.polarity = polarity
        self.embedding = None
        self.spacy_pos_stts = spacy_pos_stts
        self.spacy_pos_universal_google = spacy_pos_universal_google
        self.spacy_ner_type = spacy_ner_type
        self.spacy_ner_iob = spacy_ner_iob
        self.spacy_shape = spacy_shape
        self.spacy_is_punct = spacy_is_punct
        self.spacy_like_url = spacy_like_url
        self.spacy_like_num = spacy_like_num

    def get_key(self, text_type):
        if text_type == 'lowercase':
            return self.text.lower()
        return self.text
