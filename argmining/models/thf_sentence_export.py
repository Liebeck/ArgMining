class THFSentenceExport:
    def __init__(self, uniqueID, label, text, tokens, dependencies):
        self.uniqueID = uniqueID
        self.label = label
        self.text = text
        self.tokens = tokens
        self.dependencies = dependencies
