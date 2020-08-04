import numbers
import warnings
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import preprocessing


def is_value_numeric(value):

    """Return if the passed value is numeric"""

    return isinstance(value, numbers.Number)


def is_value_categorical(value):

    """Return if a value is categorical"""

    return type(value) == str


def round_dict_values(dictionary, decimals):

    """Round all the numeric values in the passed dictionary"""

    for key, value in dictionary.items():

        if is_value_numeric(value):

            dictionary[key] = round(value, decimals)

    return dictionary


def is_series_numeric(series):

    """Return whether the passed series is numeric (not categorical)"""

    return is_numeric_dtype(series)


def is_series_categorical(series):

    """Return whether the passed series is categorical (not numeric)"""

    return not is_numeric_dtype(series)


def get_numeric_column_names(data_frame):

    """Return the names of the numeric columns found in the passed data frame"""

    numeric_columns = list()

    for column in data_frame.columns:

        if is_series_numeric(data_frame[column]):

            numeric_columns.append(column)

    return numeric_columns


def get_categorical_column_names(data_frame):

    """Return the names of the categorical columns found in the passed data frame"""
    
    categorical_columns = list()

    for column in data_frame.columns:

        if is_series_categorical(data_frame[column]):

            categorical_columns.append(column)

    return categorical_columns


def read_data_sets(data_dir_path, data_set_names):

    """Read and return as data frames the data sets under the passed directory path that have the passed names"""

    data_frames = list()

    # parse the data sets
    for data_set_name in data_set_names:

        data_frame = pd.read_csv(data_dir_path + data_set_name + ".csv", na_values=["?"])
        data_frame.name = data_set_name
        data_frames.append(data_frame)

    return data_frames


def pre_process_data_frame(data_frame):

    """Pre-process the passed data frame"""

    # replace the missing values and normalize numeric columns
    return normalize_numeric_columns(replace_missing_values(data_frame))


def replace_missing_values(data_frame):

    """Return the passed data frame with their missing values replaced with the mean of the attribute"""

    # replace NaNs with column means
    data_frame = data_frame.apply(lambda column: fill_nan(column))

    return data_frame


def normalize_numeric_columns(data_frame):

    """ Return the passed data frame with the numeric columns normalized with values between 0 and 1"""

    for column in get_numeric_column_names(data_frame):

        data_frame[column] = preprocessing.minmax_scale(data_frame[column])

    return data_frame


def fill_nan(series):

    """Fill the NaN values in an series, with the mean (if numeric) or mode (if categorical), and return the result"""

    # ignore irrelevant warnings about NaN presence
    warnings.simplefilter("ignore")

    # use mean for numeric attributes
    if is_series_numeric(series):
        return series.fillna(series.mean())

    # use mode for categorical attributes
    return series.fillna(stats.mode(series)[0][0])


def get_class_column_name(data_frame):

    """Return the name of the class column of the passed data frame"""

    # the class is assumed to be the last column
    return data_frame.columns[-1]


def split_data_frame(data_frame, test_proportion):

    """Splits the passed data frame in training and test according to the passed proportion, separating features and class vector"""

    # separate feature matrix x and class vector y
    class_column_name = get_class_column_name(data_frame)
    y = data_frame.pop(class_column_name).to_frame()
    x = data_frame

    # perform a stratified split
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=test_proportion)

    '''data_frame_train = x_train
    data_frame_train[class_column_name] = y_train
    print(data_frame_train.to_string())
    data_frame_test = x_test
    data_frame_test[class_column_name] = y_test
    print(data_frame_test.to_string())'''

    return x_train, y_train.squeeze(), x_test, y_test.squeeze()


def stratified_fold_split_data_frame(data_frame, fold_num):

    """Splits the passed data frame in training and test for the passed number of folds, and separating features and class vector"""

    # separate feature matrix x and class vector y
    class_column_name = get_class_column_name(data_frame)
    y = data_frame.pop(class_column_name).to_frame()
    x = data_frame

    # set up the fold generator
    fold_generator = StratifiedKFold(n_splits=fold_num, shuffle=True)

    x_train_folds = list()
    y_train_folds = list()
    x_test_folds = list()
    y_test_folds = list()

    # for each fold, separate the features and class labels
    for train_indices, test_indices in fold_generator.split(x, y):

        x_train_folds.append(x.iloc[train_indices])
        y_train_folds.append(y.iloc[train_indices].squeeze())
        x_test_folds.append(x.iloc[test_indices])
        y_test_folds.append(y.iloc[test_indices].squeeze())

    return x_train_folds, y_train_folds, x_test_folds, y_test_folds
