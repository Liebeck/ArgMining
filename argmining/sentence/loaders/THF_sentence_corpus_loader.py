from argmining.models.thf_sentence_export import THFSentenceExport
from argmining.models.token import Token
import logging
import xml.etree.ElementTree as ET

logger = logging.getLogger()


# def parse_tree_tagger_lemma(node):


# Todo: Write unit tests for both cases


def load(file_path='data/THF/sentence/subtask_A_training.xml'):
    """
    Loads the THF corpus from an XML file
    :param file_path: relative path to the XML file
    :return:
    """
    logger.debug(u'Parsing XML File: {}'.format(file_path))
    sentences = []
    root = ET.parse(file_path).getroot()

    for sentence in root:
        print(sentence.tag)
        unique_ID = sentence.find("UniqueID").text
        label = sentence.find("Label").text
        text = sentence.find("Text").text
        xml_tokens = sentence.find("NLP").find("Sentences").find("Sentence").find("Tokens")
        xml_dependencies = sentence.find("NLP").find("Sentences").find("Sentence").find("Dependencies")
        tokens = []
        for xml_token in xml_tokens.getchildren():
            tree_tagger_lemma_node = xml_token.find("TreeTaggerLemma").text
            tree_tagger_lemma = None
            iwnlp_lemma_node = xml_token.find("IWNLPLemma").text
            iwnlp_lemma = None
            polarity_node = xml_token.find("Polarity").text
            polarity = None
            token_model = Token(xml_token.find("TokenIndexInSentence").text,
                                xml_token.find("Text").text,
                                xml_token.find("POSTag").text,
                                xml_token.find("MateToolsPPOS").text,
                                xml_token.find("MateToolsPLemma").text,
                                tree_tagger_lemma,
                                iwnlp_lemma,
                                polarity)
        # print(item.tag)
        #    def __init__(self, token_index_in_sentence, text, pos_tag, mate_tools_pos_tag, mate_tools_lemma, tree_tagger_lemma,iwnlp_lemma, polarity):

        # print(nlp)
        # children = sentence.getchildren()
        # print(sentence.find("UniqueID").text)
        # print(sentence.getchildren().find("entry")['UniqueID'])
        # print(ET.tostring(ET.SubElement(sentence, 'UniqueID')))
        # print(ET.SubElement(sentence, 'UniqueID').attrib)
        # print(ET.SubElement(sentence, 'UniqueID').text)
        # with open(absolute_path) as fp:
        # xml = '\n'.join(fp.readlines())
        # soup = BeautifulSoup(xml, 'xml')
        # print(soup)


        sentence_model = THFSentenceExport(unique_ID, label, text, tokens)
        sentences.append(sentence_model)

    logger.info('Parsed {} sentences'.format(len(sentences)))
