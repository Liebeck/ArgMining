from pandas_confusion import ConfusionMatrix


def write_prediction_file(path, test_unique_ids, Y_test_indices, y_prediction_classes):
    with open(path, 'w') as prediction_handler:
        prediction_handler.write('{}\t{}\t{}\n'.format("UniqueID", "Gold_Label", "Prediction"))
        for index, val in enumerate(test_unique_ids):
            prediction_handler.write('{}\t{}\t{}\n'.format(val, Y_test_indices[index], y_prediction_classes[index]))


def write_score_file(score_file, f1_mean, f1, model, Y_test_indices, y_prediction_classes):
    with open(score_file, 'w') as score_handler:
        score_handler.write("Micro-averaged F1: {}\n".format(f1_mean))
        score_handler.write("Individual scores: {}\n".format(f1))
        score_handler.write("Confusion matrix:\n")
        score_handler.write(str(ConfusionMatrix(Y_test_indices, y_prediction_classes)))
        score_handler.write("\n\n\nModel summary\n")
        model.summary(print_fn=lambda x: score_handler.write(x + '\n'))
