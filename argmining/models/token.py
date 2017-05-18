class Token:
    def __init__(self, token_index_in_sentence, text, pos_tag=None, mate_tools_pos_tag=None, mate_tools_lemma=None,
                 tree_tagger_lemma=None,
                 iwnlp_lemma=None, polarity=None, pos_spacy_stts=None, pos_spacy_uts=None):
        self.token_index_in_sentence = token_index_in_sentence
        self.text = text
        self.pos_tag = pos_tag
        self.mate_tools_pos_tag = mate_tools_pos_tag
        self.mate_tools_lemma = mate_tools_lemma
        self.tree_tagger_lemma = tree_tagger_lemma
        self.iwnlp_lemma = iwnlp_lemma
        self.polarity = polarity
        self.embedding = None
        self.pos_spacy_stts = pos_spacy_stts
        self.pos_spacy_uts = pos_spacy_uts

    def get_key(self, text_type):
        if text_type == 'lowercase':
            return self.text.lower()
        return self.text
