from operator import itemgetter
import numpy as np
import logging


def report(grid_scores, n_top=3):
    logger = logging.getLogger()
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        logger.info("Model with rank: {0}".format(i + 1))
        logger.info("Mean validation score: {0:.3f} (std: {1:.3f})".format(
            score.mean_validation_score,
            np.std(score.cv_validation_scores)))
        logger.info("Parameters: {0}".format(score.parameters))
        logger.info("")
