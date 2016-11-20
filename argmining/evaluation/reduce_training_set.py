import logging

def reduce_training_set(X_train, y_train, training_size):
    logger = logging.getLogger()
    if training_size == 100:
        return X_train, y_train
    training_size = float(training_size) / 100  # get percentage
    logger.debug('Reducing training size: {}'.format(training_size))
    row_count = len(X_train)
    logger.debug('Row count before reduction: {}'.format(row_count))
    row_upper_index = int(round(row_count * training_size))
    logger.debug('Row_upper_index: {}'.format(row_upper_index))
    X_reduced = X_train[0: row_upper_index]
    y_reduced = y_train[0: row_upper_index]
    logger.info("Training size {} => reducing {} rows to [0:{}]".format(training_size, row_count, row_upper_index))
    return X_reduced, y_reduced
