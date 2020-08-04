import abc
from copy import copy
import numpy as np
from data_functions import *


class Condition(object):

    """Class representing a rule condition"""

    def __init__(self, attribute):

        """Constructor"""

        self._attribute = attribute

    @abc.abstractmethod
    def __eq__(self, other):

        """Equality check"""

        return self._attribute == other.get_attribute()

    @abc.abstractmethod
    def __ne__(self, other):

        """Inequality check"""

        return not self.__eq__(other)

    @abc.abstractmethod
    def __hash__(self):

        """Hash function"""

        return hash(self._attribute)

    @abc.abstractmethod
    def __str__(self):

        """To string"""

        return

    @abc.abstractmethod
    def __copy__(self):

        """Return a shallow copy"""

        return

    def get_attribute(self):

        """Return the attribute"""

        return self._attribute

    @abc.abstractmethod
    def is_respected(self, attribute_value):

        """Evaluate the condition given that the attribute is given the passed value, i.e. if the condition is respected with such value"""

        return


class NumericCondition(Condition):

    """Class representing a numeric condition, expressed as an allowed range for an attribute"""

    def __init__(self, attribute, lower_bound, upper_bound):

        super().__init__(attribute)
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

    def __eq__(self, other):

        """Equality check"""

        return super().__eq__(other) and self._lower_bound == other.get_lower_bound() and self._upper_bound == other.get_upper_bound()

    def __ne__(self, other):

        """Inequality check"""

        return not self.__eq__(other)

    def __hash__(self):

        """Hash function"""

        return super().__hash__() + hash((self._lower_bound, self._upper_bound))

    def __str__(self):

        """To string"""

        if self._lower_bound == self._upper_bound:

            return "(" + str(self._attribute) + " == " + str(round(self._lower_bound, 3)) + ")"

        return "(" + str(round(self._lower_bound, 3)) + " <= " + str(self._attribute) + " <= " + str(round(self._upper_bound, 3)) + ")"

    def __copy__(self):

        """Return a shallow copy"""

        return NumericCondition(self._attribute, self._lower_bound, self._upper_bound)

    def get_lower_bound(self):

        """Return the lower bound"""

        return self._lower_bound

    def get_upper_bound(self):

        """Return the upper bound"""

        return self._upper_bound

    def is_respected(self, attribute_value):

        """Evaluate the condition given that the attribute is given the passed value, i.e. if the condition is respected with such value"""

        return self._lower_bound <= attribute_value <= self._upper_bound


class CategoricalCondition(Condition):

    """Class representing a categorical condition, expressed as a category equality for an attribute"""

    def __init__(self, attribute, category):

        super().__init__(attribute)
        self._category = category

    def __eq__(self, other):

        """Equality check"""

        return super().__eq__(other) and self._category == other.get_category()

    def __ne__(self, other):

        """Inequality check"""

        return not self.__eq__(other)

    def __hash__(self):

        """Hash function"""

        return super().__hash__() + hash(self._category)

    def __str__(self):

        """To string"""

        return "(" + self._attribute + " == " + self._category + ")"

    def __copy__(self):

        """Return a shallow copy"""

        return CategoricalCondition(self._attribute, self._category)

    def get_category(self):

        """Return the category"""

        return self._category

    def is_respected(self, attribute_value):

        """Evaluate the condition given that the attribute is given the passed value, i.e. if the condition is respected with such value"""

        return attribute_value == self._category


class Conclusion(object):

    """Class representing a rule's conclusion"""

    def __init__(self, field, value):

        """Constructor"""

        self._field = field
        self._value = value

    def __eq__(self, other):

        """Equality check"""

        return self._field == other.get_field() and self._value == other.get_value()

    def __ne__(self, other):

        """Inequality check"""

        return not self.__eq__(other)

    def __hash__(self):

        """Hash function"""

        return hash((self._field, self._value))

    def __str__(self):

        """To string"""

        return self._field + " == " + self._value

    def get_field(self):

        """Return the field"""

        return self._field

    def get_value(self):

        """Return the value"""

        return self._value


class Rule(object):

    """Class representing a classifier rule, that links a set of conditions with a conclusion"""

    def __init__(self, conditions, conclusion):

        """Constructor"""

        self._conditions = {condition.get_attribute(): condition for condition in conditions}
        self._conclusion = conclusion

    def __eq__(self, other):

        """Equality check"""

        return self._conditions == other.get_conditions() and self._conclusion == other.get_conclusion()

    def __ne__(self, other):

        """Inequality check"""

        return not self.__eq__(other)

    def __hash__(self):

        """Hash function"""

        return hash((tuple(self._conditions.values()), self._conclusion))

    def __str__(self):

        """To string"""

        string = ""

        is_first = True

        for condition in self._conditions.values():

            if not is_first:
                string += " && "
            is_first = False

            string += str(condition)

        string += " => (" + str(self._conclusion) + ")"

        return string

    def get_conditions(self):

        """Return the conditions"""

        return self._conditions

    def get_conclusion(self):

        """Return the conclusion"""

        return self._conclusion

    def get_distance_to_instance(self, instance, categorical_class_probability_dict, exponent=2):

        """Compute and return the distance between the rule and the passed instance"""

        dist = 0

        # distance is non-zero for attributes for which the rule has a condition
        for attribute, condition in self._conditions.items():

            value = instance.loc[attribute]

            # use the simplified value difference metric as categorical distance
            if is_value_categorical(value):
                dist += simplified_value_difference(attribute, value, condition.get_category(), categorical_class_probability_dict) ** exponent

            # use a numeric distance otherwise
            else:
                dist += normalized_numeric_range_distance(value, condition.get_lower_bound(), condition.get_upper_bound()) ** exponent

        return dist

    def is_instance_covered(self, instance):

        """Return whether the passed instance is covered by the rule, i.e. it meets all the conditions of the rules"""

        # the instance must respect all the conditions of the rule to be covered
        for attribute, condition in self._conditions.items():

            if not condition.is_respected(instance[attribute]):

                return False

        return True

    def get_nearest_non_covered_instance(self, instances, categorical_class_probability_dict):

        """Return the nearest instance to the rule that is not covered by it (fully compatible)"""

        min_dist = np.inf
        nearest_non_covered_instance = None

        # find the non-covered instance with minimum distance to the rule
        for _, instance in instances.iterrows():

            if not self.is_instance_covered(instance):

                dist = self.get_distance_to_instance(instance, categorical_class_probability_dict)

                if dist < min_dist:
                    min_dist = dist
                    nearest_non_covered_instance = instance

        return nearest_non_covered_instance

    def get_most_specific_generalization(self, instance):

        """Return an adapted version of the rule that generalizes to cover the passed instance, assumed to be not covered yet"""

        adapted_conditions = list()

        # check every condition of the current rule
        for attribute, condition in self._conditions.items():

            instance_value = instance[attribute]

            # respected conditions are preserved as they are
            if condition.is_respected(instance_value):

                adapted_conditions.append(copy(condition))

            # the rest of conditions are ignored (categorical, being inequalities) or adapted (numeric, requiring bound modification)
            else:

                if is_value_numeric(instance_value):

                    if instance_value < condition.get_lower_bound():
                        lower_bound = instance_value
                        upper_bound = condition.get_upper_bound()

                    else:
                        lower_bound = condition.get_lower_bound()
                        upper_bound = instance_value

                    adapted_conditions.append(NumericCondition(condition.get_attribute(), lower_bound, upper_bound))

        adapted_rule = Rule(adapted_conditions, self._conclusion)

        return adapted_rule


class RuleCoverageAndAccuracy(object):

    """Class representing three elements: a rule, its coverage and its accuracy"""

    def __init__(self, rule, coverage, accuracy):

        """Constructor"""

        self._rule = rule
        self._coverage = coverage
        self._accuracy = accuracy

    def __str__(self):

        """To string"""

        return str(self._rule) + "\t{coverage: " + str(round(self._coverage, 3)) + ", accuracy: " + str(round(self._accuracy, 3)) + "}"

    def get_rule(self):

        """Return the rule"""

        return self._rule

    def get_coverage(self):

        """Return the coverage"""

        return self._coverage

    def get_accuracy(self):

        """Return the accuracy"""

        return self._accuracy


class CorrectBoolAndDistance(object):

    """Class representing a pair of values: a boolean (correct or incorrect) and a distance"""

    def __init__(self, is_correct, dist):

        """Constructor"""

        self._is_correct = is_correct
        self._dist = dist

    def is_correct(self):

        """Return whether it is correct"""

        return self._is_correct

    def set_is_correct(self, is_correct):

        """Set whether it is correct"""

        self._is_correct = is_correct

    def get_dist(self):

        """Return the distance"""

        return self._dist

    def set_dist(self, dist):

        """Set the distance"""

        self._dist = dist


def simplified_value_difference(attribute, category0, category1, categorical_class_probability_dict, exponent=1):

    """Return the simplified value difference between the two passed categories, checking the passed dictionary of class-given-category probabilities"""

    # equal categories have no distance
    if category0 == category1:
        return 0

    # get the probability dictionaries for each category
    category0_dict = categorical_class_probability_dict[attribute].get(category0, None)
    category1_dict = categorical_class_probability_dict[attribute].get(category1, None)

    # consider the distance is maximum if there is no information about one of the categories
    if not category0_dict or not category1_dict:
        return 1

    dist = 0

    # compute the distance for each class, computing the probability difference
    for y_class in category0_dict.keys():

        category0_prob = category0_dict[y_class]
        category1_prob = category1_dict[y_class]

        dist += abs(category0_prob - category1_prob) ** exponent

    return dist


def normalized_numeric_range_distance(value, lower_bound, upper_bound):

    """Return a numeric distance considering the passed value and lower and upper bounds"""

    # the distance is zero if the value is in the bounds' range
    if lower_bound <= value <= upper_bound:
        return 0

    # otherwise, if the bounds are same, the distance is maximum
    if lower_bound == upper_bound:
        return 1

    # otherwise, measure the distance to the surpassed bound, normalizing by the range length
    if value > upper_bound:
        return (value - upper_bound) / (upper_bound - lower_bound)
    return (lower_bound - value) / (upper_bound - lower_bound)


def create_rule_from_instance(instance, class_field, class_value):

    """Create and return a rule based on the passed instance"""

    conditions = list()

    # for each field, make a condition equal to the instance value
    for field, value in instance.to_dict().items():
        
        # categorical fields have equality-based conditions
        if is_value_categorical(value):
            conditions.append(CategoricalCondition(field, value))
            
        # numeric fields have conditions expressed as a degenerate range (lower and upper bounds are equal)
        else:
            conditions.append(NumericCondition(field, value, value))
            
    # add the class field-value match as conclusion
    conclusion = Conclusion(class_field, class_value)

    return Rule(conditions, conclusion)


def get_class_probabilities_for_categorical_columns(x, y):

    """Return a dictionary with each categorical field of the feature matrix x as entry, and sub-dictionaries that map each category to a probability of each class of y"""

    categorical_class_probability_dict = dict()

    # top level dict is indexed by categorical columns
    for column in get_categorical_column_names(x):

        column_dict = dict()

        # second level is for the column's categories
        for category in x[column].unique():

            category_dict = dict()

            category_series = y[x[column] == category]

            total_count = category_series.count()

            # last level associates the category with a probability for each class of y
            for y_class in y.T.squeeze().unique():

                category_dict[y_class] = category_series[category_series == y_class].count() / total_count

            column_dict[category] = category_dict

        categorical_class_probability_dict[column] = column_dict

    return categorical_class_probability_dict


def get_nearest_rule(rule_set, instance, categorical_class_probability_dict, rule_to_ignore=None, rule_must_cover_instance=False):

    """Return the rule from the passed rule set that is more similar to the passed instance"""

    min_dist = np.inf
    nearest_rule = None

    # find the rule with minimum distance to the instance
    for rule in rule_set:

        if not rule_to_ignore or rule != rule_to_ignore:

            if not rule_must_cover_instance or rule.is_instance_covered(instance):

                dist = rule.get_distance_to_instance(instance, categorical_class_probability_dict)

                if dist < min_dist:
                    min_dist = dist
                    nearest_rule = rule

    return nearest_rule, min_dist


def get_rule_set_accuracy(rule_set, instances, class_values, class_field, categorical_class_probability_dict, use_leave_one_out, rule_must_cover_instance=False):

    """Calculate and return the accuracy of the passed rule set for the instances defined by the feature matrix x and the class vector y"""

    correct_classification_num = 0

    assigned_classes = list()

    # dictionary that will associate each instance index with the classification result (correct or not) and the distance to its nearest rule
    instance_result_dict = dict()

    # classify each instance using its nearest rule, leaving out the instance-based rule
    for index, instance in instances.iterrows():

        instance_rule = create_rule_from_instance(instance, class_field, class_values[index])

        # ignore the instance's own rule if necessary
        rule_to_ignore = None
        if use_leave_one_out:
            rule_to_ignore = instance_rule

        nearest_rule, nearest_dist = get_nearest_rule(rule_set, instance, categorical_class_probability_dict, rule_to_ignore, rule_must_cover_instance)

        if nearest_rule:

            assigned_class = nearest_rule.get_conclusion().get_value()
            assigned_classes.append(assigned_class)

            is_correct = assigned_class == class_values[index]
            if is_correct:
                correct_classification_num += 1

        else:
            is_correct = False

        instance_result_dict[index] = CorrectBoolAndDistance(is_correct, nearest_dist)

    return correct_classification_num / len(class_values), assigned_classes, instance_result_dict


def get_updated_accuracy(old_accuracy, instances, class_values, instance_result_dict, modified_rule, categorical_class_probability_dict):

    """Return an updated version of the passed accuracy taking into account the classification changes brought by the passed modified rule"""

    correct_classification_increment = 0

    new_instance_result_dict = copy(instance_result_dict)

    # for each instance, check if the new rule is nearer than the former nearest rule
    for index, instance in instances.iterrows():

        new_dist = modified_rule.get_distance_to_instance(instance, categorical_class_probability_dict)

        # if the new rule is the new nearest rule, compute if it classifies the instance correctly or not, to update the accuracy
        if new_dist < new_instance_result_dict[index].get_dist():

            is_correct = modified_rule.get_conclusion().get_value() == class_values[index]

            if is_correct != new_instance_result_dict[index].is_correct():

                if is_correct:
                    correct_classification_increment += 1

                else:
                    correct_classification_increment -= 1

            new_instance_result_dict[index] = CorrectBoolAndDistance(is_correct, new_dist)

    accuracy_increment = correct_classification_increment / len(instances)

    accuracy = old_accuracy + accuracy_increment

    return accuracy, new_instance_result_dict


def build_rise_classifier(instances, class_values, verbose=False):

    """Build a RISE rule-based/instance-based classifier using the passed feature matrix and the class vector, and return the classifier's rules"""

    # get the name of the class field
    class_field = class_values.name

    # initially, create a rule for each instance
    rule_set = {create_rule_from_instance(instance, class_field, class_values[i]) for i, instance in instances.iterrows()}

    # for the sake of faster computation, only calculate once the probabilities of each category (of all the categorical values of x) to produce each class of y
    categorical_class_probability_dict = get_class_probabilities_for_categorical_columns(instances, class_values)

    # compute the initial accuracy, and get the dictionary of results and distances that will allow later incremental updates of the accuracy
    initial_accuracy, _, instance_result_dict = get_rule_set_accuracy(rule_set, instances, class_values, class_field, categorical_class_probability_dict, True)
    accuracy = initial_accuracy
    if verbose:
        print("\t\tInitial accuracy:", round(accuracy, 3))

    has_accuracy_increased = True

    # keep on generalizing the rule set as long as it is possible to increase the accuracy
    while has_accuracy_increased:

        # examine all rules
        for rule in list(rule_set):

            # get the nearest instance to the rule that is not covered by it (fully compatible)
            nearest_non_covered_instance = rule.get_nearest_non_covered_instance(instances, categorical_class_probability_dict)

            # adapt the rule to cover the instance
            adapted_rule = rule.get_most_specific_generalization(nearest_non_covered_instance)

            # if accuracy does not get lower, use an alternative set of rules changing the old rule with the adapted one
            alternative_accuracy, alternative_instance_result_dict = get_updated_accuracy(accuracy, instances, class_values, instance_result_dict, adapted_rule, categorical_class_probability_dict)
            if alternative_accuracy >= accuracy:
                rule_set.discard(rule)
                rule_set.add(adapted_rule)
                accuracy = alternative_accuracy
                instance_result_dict = alternative_instance_result_dict

        # update the accuracy, and check if it has increased
        has_accuracy_increased = accuracy > initial_accuracy
        initial_accuracy = accuracy
        if verbose:
            print("\t\tIteration accuracy:", round(accuracy, 3))

    # find the coverage and accuracy of each rule
    rule_set_with_coverage_and_accuracy = get_rule_set_with_coverage_and_accuracy(rule_set, instances, class_values)

    return rule_set_with_coverage_and_accuracy, categorical_class_probability_dict


def get_rule_set_with_coverage_and_accuracy(rule_set, instances, class_values):

    """Return the passed rule set along with the coverage and accuracy of each of them"""

    rule_set_with_coverage_and_accuracy = set()

    for rule in rule_set:

        covered_instance_count = 0
        correct_classification_count = 0

        for index, instance in instances.iterrows():

            if rule.is_instance_covered(instance):

                covered_instance_count += 1

                if rule.get_conclusion().get_value() == class_values[index]:

                    correct_classification_count += 1

        # coverage is the proportion of instances covered by the rule
        coverage = covered_instance_count / len(instances)

        # accuracy is the proportion of correctly classified instances, by the total number of attempts
        accuracy = correct_classification_count / covered_instance_count

        rule_set_with_coverage_and_accuracy.add(RuleCoverageAndAccuracy(rule, coverage, accuracy))

    return rule_set_with_coverage_and_accuracy


def evaluate_rule_set(rule_set, instances, class_values, categorical_class_probability_dict):

    """Predict the class value of the passed instances with the passed rule set, and compare with the ground truth values to assess the accuracy"""

    # get the name of the class field
    class_field = class_values.name

    # compute the test accuracy
    accuracy, assigned_classes, _ = get_rule_set_accuracy(rule_set, instances, class_values, class_field, categorical_class_probability_dict, False)

    return accuracy, assigned_classes
