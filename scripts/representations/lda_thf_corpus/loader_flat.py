import logging
import json


class LoaderFlat(object):
    def __init__(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

    def load_thf_corpus_flat(self, path):
        self.logger.info('Loading THF corpus from {}'.format(path))
        with open(path, encoding='utf-8') as data_file:
            data = json.load(data_file)
            proposals = data["instance"]["tempelhofer-feld"]["proposals"]
            proposals_flattened = []
            comment_counter = 0
            for proposal_id, proposal in proposals.items():
                proposal['flat_comments'] = self.flatten_comments(proposal["comments"])
                proposal.pop('comments', None)
                proposal.pop('creator_badges', None)
                proposal.pop('badges', None)
                proposal.pop('tags', None)
                comment_counter += len(proposal['flat_comments'])
                proposals_flattened.append(proposal)
            self.logger.info('Loaded {} proposals with {} comments'.format(len(proposals), comment_counter))
            return proposals_flattened

    def flatten_comments(self, comments):
        all_comments = []
        for comment_id, comment in comments.items():
            subcomments = comment["comments"]
            comment['comment_id'] = comment_id
            comment.pop('comments', None)
            all_comments.append(comment)
            all_comments.extend(self.flatten_comments(subcomments))
        return all_comments
