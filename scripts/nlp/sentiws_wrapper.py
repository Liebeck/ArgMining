import logging
import io

logger = logging.getLogger()


class SentiWSEntry(object):
    def __init__(self, form, pos):
        self.form = form
        self.pos = pos


class SentiWSWrapper(object):
    def __init__(self, sentiws_path='data/sentiws'):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        self.load_files(sentiws_path)

    def load_files(self, directory):
        self.entries = {}
        self.entries.update(self.load_file(directory + '/SentiWS_v1.8c_Positive.txt'))
        self.entries.update(self.load_file(directory + '/SentiWS_v1.8c_Negative.txt'))
        self.logger.debug('SentiWS: Loaded {} entries'.format(len(self.entries)))

    def process_line(self, line):
        entries = {}
        elements = line.split("\t")
        first_entry = elements[0].split("|")
        pos = first_entry[1]
        score = float(elements[1])
        forms = []
        forms.append(first_entry[0])
        if len(elements) > 2:
            for form in elements[2].split(","):
                forms.append(form)
        for form in forms:
            entries[SentiWSEntry(form, pos)] = score
        return entries

    def load_file(self, path):
        entries = {}
        with io.open(path, encoding='utf-8') as file:
            content = file.readlines()
            content = [x.strip() for x in content]
            for line in content:
                entries.update(self.process_line(line))
        return entries
