import argparse
import logging
from scripts.representations.lda_thf_corpus.loader_flat import LoaderFlat
import spacy
import json

logger = logging.getLogger()


def config_argparser():
    argparser = argparse.ArgumentParser(description='ArgMining')
    argparser.add_argument('-corpuspath', type=str, default='data/THF/public_Tempelhofer-Feld-2015-07-07T12_41_08.json',
                           help='Name of the subtask')
    argparser.add_argument('-hilbert', dest='hilbert', action='store_true')
    argparser.set_defaults(hilbert=False)
    return argparser.parse_args()


def extract_tokens(document, spacy_pipeline):
    tokens = []
    for i, token in enumerate(spacy_pipeline(document)):
        if not token.is_punct and not token.is_space and '\r\n' not in token.text:
            tokens.append(token.text)
    return tokens


def tokens_to_json(tokens):
    tokens = []
    for token in tokens:
        tokens.append({'Text': token.text,
                       'POS': token.pos_,
                       'NER_type': token._ner_type,
                       'NER_IOB': token.ner_iob
                       })
    return tokens


def load_thf_corpus_tokenized(path, spacy_pipeline):
    documents = []
    thf_loader = LoaderFlat()
    proposals = thf_loader.load_thf_corpus_flat(path)
    for proposal in proposals:
        proposal_text = '{} {}'.format(proposal["title"], proposal["description"])
        documents.append({'Text': proposal_text,
                          'Tokens': tokens_to_json(extract_tokens(proposal_text, spacy_pipeline))
                          })
        for comment in proposal["flat_comments"]:
            documents.append({'Text': comment["text"],
                              'Tokens': tokens_to_json(extract_tokens(comment["text"], spacy_pipeline))
                              })
    logger.info('Tokenized {} documents'.format(len(documents)))
    return documents


if __name__ == '__main__':
    arguments = config_argparser()
    logger.info('Loading Spacy model')
    nlp = spacy.load('de')
    logger.info('Spacy model loaded')
    tokenized_documents = load_thf_corpus_tokenized(arguments.corpuspath, nlp)
    output_path = 'data/THF/corpus_tokenized-2015-07-07.json'
    logger.info('Writing tokenized corpus to: {}'.format(output_path))
    with open(output_path, 'w') as outfile:
        json.dump(tokenized_documents, outfile, indent=2)
