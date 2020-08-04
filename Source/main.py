from pathlib import Path
from rise_classifier import *

# printing modifiers
PRINT_BOLD_COLOR = "\033[1m"
PRINT_DISABLE_COLOR = "\033[0m"


def clear_file(file_name):

    """Clear the contents of a file"""

    # clear any previous content in the file
    open(file_name, "w").close()


def print_and_write(message, file_name):

    """Print the passed string in the console and write it to the current global file"""

    print(message)
    file = open(file_name, "a")
    file.write(message + "\n")
    file.close()


def main():

    """ Main function"""

    # show the title
    print(PRINT_BOLD_COLOR + "Rule Induction from a Set of Exemplars (RISE) -- Rule-based/Instance-based Classifier" + PRINT_DISABLE_COLOR + "\n")

    # path to the data sets
    parent_dir_path = str(Path(__file__).parents[1])
    data_dir_path = parent_dir_path + "/Data/"
    results_dir_path = parent_dir_path + "/Results/"

    # data sets to use
    data_set_names = ["contact-lenses", "labor", "hepatitis", "breast-cancer"]

    # read the data sets as data frames
    data_frames = read_data_sets(data_dir_path, data_set_names)

    # pre-process the data sets
    data_frames = [pre_process_data_frame(data_frame) for data_frame in data_frames]

    for data_set_name, data_frame in zip(data_set_names, data_frames):

        # prepare the file where to write the results
        data_set_file_name = results_dir_path + data_set_name + ".txt"
        clear_file(data_set_file_name)

        print_and_write("Data set: {}".format(data_set_name), data_set_file_name)

        # number of times to learn the model with different training-test folds
        cross_valid_num = min(10, len(data_frame) // 5 + 1)

        # split the data in folds
        x_train_folds, y_train_folds, x_test_folds, y_test_folds = stratified_fold_split_data_frame(data_frame, cross_valid_num)

        accuracies = list()

        # learn the model and evaluate it with each of the different cross-validation folds
        for index, (x_train, y_train, x_test, y_test) in enumerate(zip(x_train_folds, y_train_folds, x_test_folds, y_test_folds)):

            print_and_write("\tCV Training model #{}".format(index+1), data_set_file_name)

            # use the training set to build the classifier and produce the model's rules
            rule_set_with_coverage_and_accuracy, categorical_class_probability_dict = build_rise_classifier(x_train, y_train, True)

            # show the rules
            print_and_write("\t\tRule set:".format(len(rule_set_with_coverage_and_accuracy)), data_set_file_name)
            for i, rule_with_coverage_and_accuracy in enumerate(rule_set_with_coverage_and_accuracy):
                print_and_write("\t\t\t({}) {}".format(i+1, rule_with_coverage_and_accuracy), data_set_file_name)

            rule_set = {rule_with_coverage_and_accuracy.get_rule() for rule_with_coverage_and_accuracy in rule_set_with_coverage_and_accuracy}

            # evaluate the rules on the test set to assess the classifier's accuracy
            accuracy, assigned_classes = evaluate_rule_set(rule_set, x_test, y_test, categorical_class_probability_dict)
            accuracies.append(accuracy)

            # show the assigned classes for each test instance
            print_and_write("\t\tTest instances with assigned classes:", data_set_file_name)
            for i, ((_, instance), real_class, assigned_class) in enumerate(zip(x_test.iterrows(), y_test, assigned_classes)):
                print_and_write("\t\t\t({}) Attributes: {},\tReal class: {},\tPredicted class: {}".format(i+1, round_dict_values(instance.to_dict(), 3), real_class, assigned_class), data_set_file_name)

            print_and_write("\t\tTest accuracy: {}".format(round(accuracy, 3)), data_set_file_name)

        # compute the mean accuracy
        mean_accuracy = float(np.mean(accuracies))
        accuracy_std = float(np.std(accuracies))
        print_and_write("\tCV Mean accuracy: {} ± {}\n\n".format(round(mean_accuracy, 3), round(accuracy_std, 3)), data_set_file_name)

        # save the accuracy information in a separate tabular file
        accuracy_dict = {"CV-{}".format(i+1): accuracy for i, accuracy in enumerate(accuracies)}
        accuracy_dict["Avg"] = mean_accuracy
        accuracy_dict["Std"] = accuracy_std
        accuracy_file_path = results_dir_path + data_set_name + "_accuracy.xlsx"
        clear_file(accuracy_file_path)
        writer = pd.ExcelWriter(accuracy_file_path)
        pd.Series(accuracy_dict).to_frame("Accuracy").to_excel(writer, 'Matrix')
        writer.save()


if __name__ == "__main__":
    main()
