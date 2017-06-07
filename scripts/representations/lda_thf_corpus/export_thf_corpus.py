import argparse
import json
import logging
import spacy
from scripts.nlp.spacy_wrapper import SpacyWrapper
from scripts.representations.lda_thf_corpus.loader_flat import LoaderFlat

logger = logging.getLogger()


def config_argparser():
    argparser = argparse.ArgumentParser(description='ArgMining')
    argparser.add_argument('-corpuspath', type=str, default='data/THF/public_Tempelhofer-Feld-2015-07-07T12_41_08.json',
                           help='Name of the subtask')
    argparser.add_argument('-hilbert', dest='hilbert', action='store_true')
    argparser.set_defaults(hilbert=False)
    return argparser.parse_args()


def extract_tokens(document, spacy):
    tokens = []
    for i, token in enumerate(spacy.process_sentence(document)['tokens']):
        if not token.spacy_is_punct and not token.spacy_is_space and '\r\n' not in token.text:
            tokens.append(token)
    return tokens


def tokens_to_json(tokens):
    json_tokens = []
    for token in tokens:
        json_tokens.append({'Text': token.text,
                            'Lemma': token.iwnlp_lemma,
                            'POS': token.spacy_pos_universal_google,
                            'NER_type': token.spacy_ner_type,
                            'NER_IOB': token.spacy_ner_iob
                            })
    return json_tokens


def load_thf_corpus_tokenized(path, spacy):
    documents = []
    thf_loader = LoaderFlat()
    proposals = thf_loader.load_thf_corpus_flat(path)
    for proposal in proposals:
        proposal_text = '{} {}'.format(proposal["title"], proposal["description"])
        documents.append({'Text': proposal_text,
                          'Tokens': tokens_to_json(extract_tokens(proposal_text, spacy))
                          })
        for comment in proposal["flat_comments"]:
            documents.append({'Text': comment["text"],
                              'Tokens': tokens_to_json(extract_tokens(comment["text"], spacy))
                              })
    logger.info('Tokenized {} documents'.format(len(documents)))
    return documents


if __name__ == '__main__':
    arguments = config_argparser()
    spacy = SpacyWrapper()
    tokenized_documents = load_thf_corpus_tokenized(arguments.corpuspath, spacy)
    output_path = 'data/THF/corpus_tokenized-2015-07-07.json'
    logger.info('Writing tokenized corpus to: {}'.format(output_path))
    with open(output_path, 'w') as outfile:
        json.dump(tokenized_documents, outfile, indent=2)
