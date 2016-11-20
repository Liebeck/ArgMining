import logging


def reduce_training_set(X_train, y_train, training_size):
    logger = logging.getLogger()
    logger.debug('Reducing training size: {}'.format(training_size))
    if training_size == 100:
        return X_train, y_train
    training_size = float(training_size) / 100  # get percentage
    row_count = X_train.shape[0]
    logger.debug('Row count before reduction: {}'.format(row_count))
    row_upper_index = int(round(row_count * training_size))
    logger.debug('Row_upper_index: {}'.format(row_upper_index))
    row_indices = range(0, row_upper_index)
    X_reduced = X_train[row_indices, :]
    y_reduced = y_train[row_indices]
    logger.info("Training size {} => reducing {} rows to [0:{}]".format(training_size, row_count, row_upper_index))
    return X_reduced, y_reduced
