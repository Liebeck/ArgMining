import logging
import spacy
import json

class LoaderFlat(object):
    def __init__(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        # self.logger.info('Loading Spacy model')
        # self.nlp = spacy.load('de')
        # self.logger.info('Spacy model loaded')

    def extract_tokens(self, document):
        return None

    def load_thf_corpus_flat(self, path):
        self.logger.info('Loading THF corpus from {}'.format(path))
        with open(path, encoding='utf-8') as data_file:
            data = json.load(data_file)
            proposals = data["instance"]["tempelhofer-feld"]["proposals"]
            tokenized_documents = []
            # self.logger.info('Processing {} proposals'.format(len(proposals)))
            all_comments = []
            for proposal_id, proposal in proposals.items():
                # self.logger.info('Processing proposal {}:'.format(proposal_id))
                proposal_text = '{} {}'.format(proposal["title"], proposal["description"])
                tokenized_documents.append(self.extract_tokens(proposal_text))
                flattened_comments = self.flatten_comments(proposal["comments"])
                for comment in flattened_comments:
                    self.logger.info(comment)
                    break
                    tokenized_documents.append(self.extract_tokens(comment["text"]))
                # self.logger.info('Proposal {} has {} comments'.format(proposal_id, len(flattened_comments)))
                all_comments.extend(flattened_comments)
                # self.logger.info(proposal_text.encode('utf-8'))
                # break
            self.logger.info('Loaded {} proposals with {} comments'.format(len(proposals), len(all_comments)))
            self.logger.info('Tokenized {} documents'.format(len(tokenized_documents)))
            return tokenized_documents

    def flatten_comments(self, comments):
        all_comments = []
        for comment_id, comment in comments.items():
            all_comments.extend(self.flatten_comments(comment["comments"]))
        all_comments.extend(comments)
        return all_comments

