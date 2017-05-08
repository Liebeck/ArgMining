import argparse
import logging
from scripts.resources.lda_thf_corpus.loader_flat import LoaderFlat
import spacy

logger = logging.getLogger()


def config_argparser():
    argparser = argparse.ArgumentParser(description='ArgMining')
    argparser.add_argument('-corpuspath', type=str, default='data/THF/public_Tempelhofer-Feld-2015-07-07T12_41_08.json',
                           help='Name of the subtask')
    argparser.add_argument('-hilbert', dest='hilbert', action='store_true')
    argparser.set_defaults(hilbert=False)
    return argparser.parse_args()


def extract_tokens(document, spacy_pipeline):
    return None


def load_thf_corpus_tokenized(path, spacy_pipeline):
    tokenized_documents = []
    thf_loader = LoaderFlat()
    proposals = thf_loader.load_thf_corpus_flat(path)
    for proposal in proposals:
        proposal_text = '{} {}'.format(proposal["title"], proposal["description"])
        tokenized_documents.append(extract_tokens(proposal_text, spacy_pipeline))
        for comment in proposal["flat_comments"]:
            tokenized_documents.append(extract_tokens(comment["text"], spacy_pipeline))

    logger.info('Tokenized {} documents'.format(len(tokenized_documents)))
    return tokenized_documents


if __name__ == '__main__':
    arguments = config_argparser()
    logger.info('Loading Spacy model')
    # nlp = spacy.load('de')
    nlp = None
    logger.info('Spacy model loaded')
    tokenized_documents = load_thf_corpus_tokenized(arguments.corpuspath, nlp)
