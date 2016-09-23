class Token:
    def __init__(self, token_index_in_sentence, text, pos_tag, mate_tools_pos_tag, mate_tools_lemma, tree_tagger_lemma,
                 iwnlp_lemma, polarity):
        self.token_index_in_sentence = token_index_in_sentence
        self.text = text
        self.pos_tag = pos_tag
        self.mate_tools_pos_tag = mate_tools_pos_tag
        self.mate_tools_lemma = mate_tools_lemma
        self.tree_tagger_lemma = tree_tagger_lemma
        self.iwnlp_lemma = iwnlp_lemma
        self.polarity = polarity