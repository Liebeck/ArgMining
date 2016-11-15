from operator import itemgetter
import logging


def report_best_results(cv_results_, n_top=4):
    logger = logging.getLogger()
    logger.info("Printing gridsearch results:")
    means = cv_results_['mean_test_score']
    stds = cv_results_['std_test_score']
    params = cv_results_['params']
    results = zip(means, stds, params)
    results = sorted(results, key=itemgetter(0), reverse=True)[:n_top]
    for i in range(0, n_top):
        mean, std, params = results[i]
        logger.info("Model with rank: {0}".format(i + 1))
        logger.info("Mean validation score: {0:.3f} (std: {1:.3f})".format(mean, std))
        logger.info("Parameters: {0}".format(params))
        if i < (n_top - 1):
            logger.info("")
