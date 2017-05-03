import argparse
import logging
from scripts.resources.lda_thf_corpus.loader_flat import load_thf_corpus_flat


def config_argparser():
    argparser = argparse.ArgumentParser(description='ArgMining')
    argparser.add_argument('-corpuspath', type=str, default='data/THF/public_Tempelhofer-Feld-2015-07-07T12_41_08.json',
                           help='Name of the subtask')
    argparser.add_argument('-hilbert', dest='hilbert', action='store_true')
    argparser.set_defaults(hilbert=False)
    return argparser.parse_args()


if __name__ == '__main__':
    arguments = config_argparser()
    logger = logging.getLogger()
    flat_corpus = load_thf_corpus_flat(arguments.corpuspath)
