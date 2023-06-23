import json
import re
from collections import Counter
from enum import Enum

import evaluate
import numpy
from fuzzywuzzy import fuzz

numpy.random.seed(42)
unanswerable = 'unanswerable'
feature_names = ["disease name", "location", "date", "%CI values", "method"]

unanswerable_string_patterns = ['not possible', 'This information is not provided in the context', 
                                'not provided', 'The article does not provide any specific values for the properties', 'not available']


class MatchType(Enum):
    EXACT = "EXACT"
    PARTIAL = "PARTIAL"

def check_string_match(input_string, match_strings):
    for mat_str in match_strings:
        match = re.search(mat_str, input_string)
        if match:
            return True
    return False
    
def is_answerable(text):
    return (str(text).strip().lower() != unanswerable) and (not check_string_match(str(text).strip().lower(), unanswerable_string_patterns))


def is_unanswerable(text):
    return (str(text).strip().lower() == unanswerable) or (check_string_match(str(text).strip().lower(), unanswerable_string_patterns))


def calculate_fuzz_ratio(text1, text2):
    return fuzz.ratio(str(text1).strip().lower(), str(text2).strip().lower())


def get_contribution_feature_values_list(contribution):
    return str(contribution).strip().split("\n")


def is_partial_match_text_based(label_feature_value, prediction_feature_value):
    return label_feature_value != '-' and prediction_feature_value != '-' and label_feature_value != '' and prediction_feature_value != '' \
        and calculate_fuzz_ratio(label_feature_value, prediction_feature_value) >= 85


def is_partial_match_json_based(label_feature_value, prediction_feature_value):
    return label_feature_value != '-' and prediction_feature_value != '-' and label_feature_value != "" and prediction_feature_value != "" \
        and calculate_fuzz_ratio(label_feature_value, prediction_feature_value) >= 85


def is_exact_match_text_based(label_feature_value, prediction_feature_value):
    return label_feature_value != '-' and prediction_feature_value != '-' \
        and label_feature_value != '' and prediction_feature_value != '' \
        and label_feature_value.strip().lower() == prediction_feature_value.strip().lower()


def is_exact_match_json_based(label_feature_value, prediction_feature_value):
    return label_feature_value != '-' and prediction_feature_value != '-' \
        and label_feature_value != "" and prediction_feature_value != "" \
        and str(label_feature_value).strip().lower() == str(prediction_feature_value).strip().lower()


def get_number_of_values_of_feature_text_based(feature_name, text):
    return sum(
        [1 for contribution in text.split("|") if has_feature_value_text_based(contribution, feature_name)])
    # print("in this text ", feature_name, " has ", number_of_features_with_value, " occurrence with values ")
    # return number_of_features_with_value


def get_number_of_values_of_feature_json_based(feature_name, json_item_list):
    number_of_features_with_value = sum(
        [1 for contribution in json_item_list if has_feature_value_json_based(contribution, feature_name)])
    # print("in this text ", feature_name, " has ", number_of_features_with_value, " occurrence with values ")
    return number_of_features_with_value


def get_r0_values_of_string(contribution):
    contribution_values = str(contribution).strip().split("\n")
    r0_value_string = next((s.lower() for s in contribution_values if "R0 value".lower() in s.lower()), "").replace(
        "R0 value:".lower(), "").strip().lower()
    return re_r0_value_extractor(r0_value_string)


def get_r0_values_of_json(contribution):
    contribution_root = get_contribution_root(contribution)
    r0_value_string = get_feature_value_json_based(contribution_root=contribution_root, feature_name="R0 value")
    return re_r0_value_extractor(r0_value_string)


def re_r0_value_extractor(text):
    return re.findall(r"\d+\.\d+|\d+", text.lower())


def has_feature_value_text_based(contribution, feature_name):
    contribution_feature_values = get_contribution_feature_values_list(contribution)
    feature_element = next((s for s in contribution_feature_values if feature_name.lower() in s.lower()), "")
    feature_value = feature_element.lower().replace(feature_name.lower() + ":", "").strip().lower()
    return feature_value != '-' and feature_value != ""


def has_feature_value_json_based(contribution, feature_name):
    contribution_root = get_contribution_root(contribution)
    feature_value = get_feature_value_json_based(contribution_root, feature_name)
    # contribution_root = contribution.get("contribution", {}) if isinstance(contribution, dict) else {}
    # feature_value = contribution_root.get(feature_name, {}) if isinstance(contribution_root, dict) else {}
    return feature_value != '-' and feature_value != ""


def get_number_of_r0_values_text_based(text):
    # x = sum([len(get_r0_values_of_string(contribution)) for contribution in contribution_list])
    # print("in this text R0_value has ", x, "values")
    return sum([len(get_r0_values_of_string(contribution)) for contribution in text.split("|")])


def get_number_of_r0_values_json_based(json_item_list):
    return sum([len(get_r0_values_of_json(contribution)) for contribution in json_item_list])


def get_feature_value_json_based(contribution_root, feature_name):
    feature_value = ""
    if isinstance(contribution_root, dict):
        for key in contribution_root.keys():
            if key.lower() == feature_name.lower():
                feature_value = str(contribution_root[key]).strip().lower()
                break
    return feature_value


def get_contribution_root(contribution):
    contribution_root = {}
    if isinstance(contribution, dict):
        contribution_keys = [key for key in contribution.keys() if key.lower() == "contribution"]
        if contribution_keys:
            contribution_root = contribution[contribution_keys[0]]

    return contribution_root


def extract_label_prediction_feature_values_text(feature_name, label_contribution, prediction_contribution):
    label_contribution_value_list = get_contribution_feature_values_list(label_contribution)
    prediction_contribution_value_list = get_contribution_feature_values_list(prediction_contribution)
    label_feature_value = next(
        (s.lower() for s in label_contribution_value_list if feature_name.lower() in s.lower()), "").replace(
        feature_name.lower() + ":", "").strip().lower()
    prediction_feature_value = next(
        (s.lower() for s in prediction_contribution_value_list if feature_name.lower() in s.lower()), "").replace(
        feature_name.lower() + ":", "").strip().lower()
    return label_feature_value, prediction_feature_value


def extract_label_prediction_feature_values_json(feature_name, label_contribution, prediction_contribution):
    label_feature_value = label_contribution.get("contribution", {}).get(feature_name, "") if isinstance(
        label_contribution, dict) else ""
    prediction_contribution_root = get_contribution_root(contribution=prediction_contribution)
    prediction_feature_value = get_feature_value_json_based(contribution_root=prediction_contribution_root,
                                                            feature_name=feature_name)
    return label_feature_value, prediction_feature_value


def compute_exact_score_between_2_contributions_text_based(prediction_contribution, label_contribution, feature_name):
    label_feature_value, prediction_feature_value = extract_label_prediction_feature_values_text(feature_name,
                                                                                                 label_contribution,
                                                                                                 prediction_contribution)
    if is_exact_match_text_based(label_feature_value, prediction_feature_value):
        return 1
    else:
        return 0


def compute_partial_score_between_2_contributions_text_based(prediction_contribution, label_contribution, feature_name):
    label_feature_value, prediction_feature_value = extract_label_prediction_feature_values_text(feature_name,
                                                                                                 label_contribution,
                                                                                                 prediction_contribution)
    if is_partial_match_text_based(label_feature_value, prediction_feature_value):
        return 1
    else:
        return 0


def compute_exact_score_between_2_contributions_json_based(prediction_contribution, label_contribution, feature_name):
    label_feature_value, prediction_feature_value = extract_label_prediction_feature_values_json(feature_name,
                                                                                                 label_contribution,
                                                                                                 prediction_contribution)

    if is_exact_match_json_based(label_feature_value=label_feature_value,
                                 prediction_feature_value=prediction_feature_value):
        return 1
    else:
        return 0


def compute_partial_score_between_2_contributions_json_based(prediction_contribution, label_contribution, feature_name):
    label_feature_value, prediction_feature_value = extract_label_prediction_feature_values_json(feature_name,
                                                                                                 label_contribution,
                                                                                                 prediction_contribution)
    if is_partial_match_json_based(label_feature_value=label_feature_value,
                                   prediction_feature_value=prediction_feature_value):
        return 1
    else:
        return 0


def get_r0_values_partial_tps_text_based(label_contribution, prediction_contribution):
    r0_values_of_label = get_r0_values_of_string(label_contribution)
    r0_values_of_prediction = get_r0_values_of_string(prediction_contribution)
    r0_values_of_label_counter = Counter(r0_values_of_label)
    r0_values_of_prediction_counter = Counter(r0_values_of_prediction)
    common_elements = (set(r0_values_of_label_counter) & set(r0_values_of_prediction_counter))
    total_score = sum([min(r0_values_of_label_counter[element], r0_values_of_prediction_counter[element]) for element in
                       common_elements])
    # print(total_score, " matches between:\n", "label: ", r0_values_of_label, "\nprediction: ", r0_values_of_prediction)
    assert total_score <= min(len(r0_values_of_label), len(r0_values_of_prediction))
    return total_score


def get_r0_values_partial_tps_json_based(label_contribution, prediction_contribution):
    r0_values_of_label = get_r0_values_of_json(label_contribution)
    r0_values_of_prediction = get_r0_values_of_json(prediction_contribution)

    r0_values_of_label_counter = Counter(r0_values_of_label)
    r0_values_of_prediction_counter = Counter(r0_values_of_prediction)

    common_elements = (set(r0_values_of_label_counter) & set(r0_values_of_prediction_counter))
    total_score = sum([min(r0_values_of_label_counter[element], r0_values_of_prediction_counter[element]) for element in
                       common_elements])
    # print(total_score, " matches between:\n", "label: ", r0_values_of_label, "\nprediction: ", r0_values_of_prediction)
    assert total_score <= min(len(r0_values_of_label), len(r0_values_of_prediction))
    return total_score


def get_contribution_list_text_based(text):
    return [contribution.strip() for contribution in text.split("|")]


def make_list_of_pairs_text_based(label_list, prediction_list):
    # make list of (label,prediction,similarity)
    list_of_label_prediction_pairs = []
    for label, prediction in zip(label_list, prediction_list):
        pair_list = []
        label_contribution_list = get_contribution_list_text_based(label)
        prediction_contribution_list = get_contribution_list_text_based(prediction)
        for item1 in label_contribution_list:
            for item2 in prediction_contribution_list:
                pair_list.append((item1, item2, calculate_fuzz_ratio(item1, item2)))

        max_selectable_pairs = min(len(label_contribution_list), len(prediction_contribution_list))
        top_similar_pairs = sorted(pair_list, key=lambda x: x[2], reverse=True)[:max_selectable_pairs]
        list_of_label_prediction_pairs.extend(top_similar_pairs)

    return list_of_label_prediction_pairs


def make_list_of_pairs_json_based(label_list, prediction_list):
    # make list of (label,prediction,similarity)
    list_of_label_prediction_pairs = []
    for label, prediction in zip(label_list, prediction_list):
        pair_list = []
        label_contribution_list = get_contribution_list_json_based(label)
        prediction_contribution_list = get_contribution_list_json_based(prediction)
        for item1 in label_contribution_list:
            for item2 in prediction_contribution_list:
                item1_str = item1
                item2_str = item2
                if isinstance(item1, dict):
                    item1_str = json.dumps(item1)
                if isinstance(item2, dict):
                    item2_str = json.dumps(item2)

                pair_list.append((item1, item2, calculate_fuzz_ratio(item1_str, item2_str)))

        max_selectable_pairs = min(len(label_contribution_list), len(prediction_contribution_list))
        top_similar_pairs = sorted(pair_list, key=lambda x: x[2], reverse=True)[:max_selectable_pairs]
        list_of_label_prediction_pairs.extend(top_similar_pairs)

    return list_of_label_prediction_pairs


def get_contribution_list_json_based(text):
    contribution_list = parse_json_string(text)
    if not isinstance(contribution_list, list):
        contribution_list = [contribution_list]

    return contribution_list


def get_feature_based_denominator_count_text_based(feature_name, item_list):
    result = 0
    for item in item_list:
        if is_answerable(item):
            result += get_number_of_values_of_feature_text_based(feature_name, item)
    return result


def get_feature_based_denominator_count_json_based(feature_name, item_list):
    result = 0
    for item in item_list:
        json_item_list = get_contribution_list_json_based(item)
        if is_answerable(json_item_list[0]):
            result += get_number_of_values_of_feature_json_based(feature_name, json_item_list)
    return result


def get_r0_partial_denominator_count_text_based(item_list):
    result = 0
    for item in item_list:
        if is_answerable(item):
            result += get_number_of_r0_values_text_based(item)
    return result


def get_r0_partial_denominator_count_json_based(item_list):
    result = 0
    for item in item_list:
        json_item_list = get_contribution_list_json_based(item)
        if is_answerable(json_item_list[0]):
            result += get_number_of_r0_values_json_based(json_item_list)

    return result


def calculate_r0_partial_tp_text_based(pair_list):
    result = 0
    for pair in pair_list:
        label_contribution, prediction_contribution = pair[0], pair[1]
        result += get_r0_values_partial_tps_text_based(label_contribution=label_contribution,
                                                       prediction_contribution=prediction_contribution)
    return result


def calculate_r0_partial_tp_json_based(pair_list):
    result = 0
    for pair in pair_list:
        label_contribution, prediction_contribution = pair[0], pair[1]
        result += get_r0_values_partial_tps_json_based(label_contribution, prediction_contribution)
    return result


def calculate_feature_based_tp_text_based(pair_list, feature_name, mode):
    result = 0
    for pair in pair_list:
        label, prediction = pair[0], pair[1]

        if mode == MatchType.EXACT.value:
            result += compute_exact_score_between_2_contributions_text_based(prediction_contribution=prediction,
                                                                             label_contribution=label,
                                                                             feature_name=feature_name)
        if mode == MatchType.PARTIAL.value:
            result += compute_partial_score_between_2_contributions_text_based(prediction_contribution=prediction,
                                                                               label_contribution=label,
                                                                               feature_name=feature_name)
    return result


def calculate_feature_based_tp_json_based(pair_list, feature_name, mode):
    result = 0
    for pair in pair_list:
        label, prediction = pair[0], pair[1]
        if mode == MatchType.EXACT.value:
            result += compute_exact_score_between_2_contributions_json_based(prediction_contribution=prediction,
                                                                             label_contribution=label,
                                                                             feature_name=feature_name)
        if mode == MatchType.PARTIAL.value:
            result += compute_partial_score_between_2_contributions_json_based(prediction_contribution=prediction,
                                                                               label_contribution=label,
                                                                               feature_name=feature_name)
    return result


def get_result_dict(exact_f1, exact_precision, exact_recalls, general_accuracy, partial_f1, partial_precision,
                    partial_recalls):
    return {"general_accuracy": general_accuracy, "exact_recalls": exact_recalls,
            "partial_recalls": partial_recalls, "exact_precisions": exact_precision,
            "partial_precisions": partial_precision, "exact_f1s": exact_f1, "partial_f1s": partial_f1}


def calculate_f1(recall, precision):
    if precision + recall != 0:
        return 2 * (precision * recall) / (precision + recall)
    else:
        return 0.0


def get_all_f1(exact_precision, exact_recalls, partial_precision, partial_recalls):
    exact_f1 = {key: calculate_f1(exact_recalls[key], exact_precision[key]) for key in exact_recalls if
                key in exact_precision}
    partial_f1 = {key: calculate_f1(partial_recalls[key], partial_precision[key]) for key in partial_recalls if
                  key in partial_precision}
    return exact_f1, partial_f1


def flatten_result_dict(evaluation_dict):
    result = {}
    for k, v in evaluation_dict.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                result[f"{k}_{sub_k}"] = sub_v
        else:
            result[k] = v
    return result


# tp / p
def recall_text_based(label_list, pair_list, feature_name, mode=MatchType.EXACT.value):
    total_items_with_value_for_feature_in_labels = get_feature_based_denominator_count_text_based(feature_name,
                                                                                                  label_list)
    total_items_with_correct_value_in_predictions = calculate_feature_based_tp_text_based(pair_list, feature_name, mode)

    if total_items_with_value_for_feature_in_labels == 0:
        return 0
    else:
        recall = total_items_with_correct_value_in_predictions / total_items_with_value_for_feature_in_labels
        assert 0 <= recall <= 1
        # print(mode + " ", feature_name, "recall: ", recall, "\ttp: ",
        #       total_items_with_correct_value_in_predictions, "\tp: ",
        #       total_items_with_value_for_feature_in_labels)
        return recall


def recall_json_based(label_list, pair_list, feature_name, mode=MatchType.EXACT.value):
    total_items_with_value_for_feature_in_labels = get_feature_based_denominator_count_json_based(feature_name,
                                                                                                  label_list)
    total_items_with_correct_value_in_predictions = calculate_feature_based_tp_json_based(pair_list, feature_name, mode)

    if total_items_with_value_for_feature_in_labels == 0:
        return 0
    else:
        recall = total_items_with_correct_value_in_predictions / total_items_with_value_for_feature_in_labels
        assert 0 <= recall <= 1
        # print(mode + " ", feature_name, "recall: ", recall, "\ttp: ",
        #       total_items_with_correct_value_in_predictions, "\tp: ",
        #       total_items_with_value_for_feature_in_labels)
        return recall


# tp/pp
def precision_text_based(prediction_list, pair_list, feature_name, mode=MatchType.EXACT.value):
    total_items_with_value_for_feature_in_predictions = get_feature_based_denominator_count_text_based(feature_name,
                                                                                                       prediction_list)
    total_items_with_correct_value_in_predictions = calculate_feature_based_tp_text_based(pair_list, feature_name, mode)

    if total_items_with_value_for_feature_in_predictions == 0:
        return 0
    else:
        precision = total_items_with_correct_value_in_predictions / total_items_with_value_for_feature_in_predictions
        assert 0 <= precision <= 1
        # print(mode + " ", feature_name, "precision: ", precision, "\ttp: ",
        #       total_items_with_correct_value_in_predictions, "\tpp: ",
        #       total_items_with_value_for_feature_in_predictions)
        return precision


# tp/pp
def precision_json_based(prediction_list, pair_list, feature_name, mode=MatchType.EXACT.value):
    total_items_with_value_for_feature_in_predictions = get_feature_based_denominator_count_json_based(feature_name,
                                                                                                       prediction_list)
    total_items_with_correct_value_in_predictions = calculate_feature_based_tp_json_based(pair_list, feature_name, mode)

    if total_items_with_value_for_feature_in_predictions == 0:
        return 0
    else:
        precision = total_items_with_correct_value_in_predictions / total_items_with_value_for_feature_in_predictions
        assert 0 <= precision <= 1
        # print(mode + " ", feature_name, "precision: ", precision, "\ttp: ",
        #       total_items_with_correct_value_in_predictions, "\tpp: ",
        #       total_items_with_value_for_feature_in_predictions)
        return precision


# tp / p
def r0_partial_recall_text_based(label_list, pair_list):
    total_decimal_r0_values_in_labels = get_r0_partial_denominator_count_text_based(label_list)
    total_correct_decimal_r0_values_in_predictions = calculate_r0_partial_tp_text_based(pair_list)

    if total_decimal_r0_values_in_labels == 0:
        return 0
    else:
        recall = total_correct_decimal_r0_values_in_predictions / total_decimal_r0_values_in_labels
        assert 0 <= recall <= 1
        # print("partial r0_value recall: ", recall, "\ttp: ",
        #       total_correct_decimal_r0_values_in_predictions, "\tp: ",
        #       total_decimal_r0_values_in_labels)
        return recall


def r0_partial_recall_json_based(label_list, pair_list):
    total_decimal_r0_values_in_labels = get_r0_partial_denominator_count_json_based(label_list)
    total_correct_decimal_r0_values_in_predictions = calculate_r0_partial_tp_json_based(pair_list)

    if total_decimal_r0_values_in_labels == 0:
        return 0
    else:
        recall = total_correct_decimal_r0_values_in_predictions / total_decimal_r0_values_in_labels
        assert 0 <= recall <= 1
        # print("partial r0_value recall: ", recall, "\ttp: ",
        #       total_correct_decimal_r0_values_in_predictions, "\tp: ",
        #       total_decimal_r0_values_in_labels)
        return recall


# tp/pp
def r0_partial_precision_text_based(prediction_list, pair_list):
    total_decimal_r0_values_in_predictions = get_r0_partial_denominator_count_text_based(prediction_list)
    total_correct_decimal_r0_values_in_predictions = calculate_r0_partial_tp_text_based(pair_list)

    if total_decimal_r0_values_in_predictions == 0:
        return 0
    else:
        precision = total_correct_decimal_r0_values_in_predictions / total_decimal_r0_values_in_predictions
        assert 0 <= precision <= 1
        # print("partial r0_value precision: ", precision, "\ttp: ",
        #       total_correct_decimal_r0_values_in_predictions, "\tp: ",
        #       total_decimal_r0_values_in_predictions)
        return precision


def r0_partial_precision_json_based(prediction_list, pair_list):
    total_decimal_r0_values_in_predictions = get_r0_partial_denominator_count_json_based(prediction_list)
    total_correct_decimal_r0_values_in_predictions = calculate_r0_partial_tp_json_based(pair_list)

    if total_decimal_r0_values_in_predictions == 0:
        return 0
    else:
        precision = total_correct_decimal_r0_values_in_predictions / total_decimal_r0_values_in_predictions
        assert 0 <= precision <= 1
        # print("partial r0_value precision: ", precision, "\ttp: ",
        #       total_correct_decimal_r0_values_in_predictions, "\tp: ",
        #       total_decimal_r0_values_in_predictions)
        return precision


def parse_json_string(data):
    try:
        parsed_json = json.loads(data)
        return parsed_json
    except json.JSONDecodeError:
        return data


def compute_all_metrics_text_based(label_list, prediction_list):
    label_prediction_pairs = make_list_of_pairs_text_based(label_list, prediction_list)
    general_accuracy = Metrics.general_accuracy_text_based(label_list, prediction_list)
    exact_recalls, partial_recalls = Metrics.all_recall_text_based(label_list, label_prediction_pairs)
    exact_precision, partial_precision = Metrics.all_precision_text_based(prediction_list, label_prediction_pairs)
    exact_f1, partial_f1 = get_all_f1(exact_precision, exact_recalls, partial_precision, partial_recalls)

    return get_result_dict(exact_f1, exact_precision, exact_recalls, general_accuracy, partial_f1,
                           partial_precision, partial_recalls)


def compute_all_metrics_json_based(label_list, prediction_list):
    label_prediction_pairs = make_list_of_pairs_json_based(label_list=label_list, prediction_list=prediction_list)
    general_accuracy = Metrics.general_accuracy_json_based(label_list=label_list, prediction_list=prediction_list)
    exact_recalls, partial_recalls = Metrics.all_recall_json_based(label_list=label_list,
                                                                   pair_list=label_prediction_pairs)

    exact_precision, partial_precision = Metrics.all_precision_json_based(prediction_list=prediction_list,
                                                                          pair_list=label_prediction_pairs)

    exact_f1, partial_f1 = get_all_f1(exact_precision, exact_recalls, partial_precision, partial_recalls)

    return get_result_dict(exact_f1, exact_precision, exact_recalls, general_accuracy, partial_f1,
                           partial_precision, partial_recalls)


class Metrics:
    @staticmethod
    # (tp+tn)/p+n
    def general_accuracy_text_based(label_list, prediction_list):
        tp = tn = 0
        for label, prediction in zip(label_list, prediction_list):
            if is_answerable(label) and is_answerable(prediction):
                tp += 1
            if is_unanswerable(label) and is_unanswerable(prediction):
                tn += 1

        accuracy = (tp + tn) / len(label_list)
        assert 0 <= accuracy <= 1
        return accuracy

    @staticmethod
    # (tp+tn)/p+n
    def general_accuracy_json_based(label_list, prediction_list):
        tp = tn = 0
        for label, prediction in zip(label_list, prediction_list):
            label = parse_json_string(label)
            prediction = parse_json_string(prediction)
            if is_answerable(label) and is_answerable(prediction):
                tp += 1
            if is_unanswerable(label) and is_unanswerable(prediction):
                tn += 1
        accuracy = (tp + tn) / len(label_list)
        assert 0 <= accuracy <= 1
        return accuracy

    @staticmethod
    def all_recall_text_based(label_list, pair_list):

        exact_results = {feature_name: recall_text_based(label_list, pair_list, feature_name, MatchType.EXACT.value) for
                         feature_name in feature_names}
        partial_results = {feature_name: recall_text_based(label_list, pair_list, feature_name, MatchType.PARTIAL.value)
                           for feature_name in feature_names}

        exact_results["R0 value"] = (recall_text_based(label_list, pair_list, "R0 value", MatchType.EXACT.value))
        partial_results["R0 value"] = (r0_partial_recall_text_based(label_list, pair_list))

        exact_results["overall"] = sum(exact_results.values()) / len(exact_results)
        partial_results["overall"] = sum(partial_results.values()) / len(partial_results)

        return exact_results, partial_results

    @staticmethod
    def all_recall_json_based(label_list, pair_list):

        exact_results = {feature_name: recall_json_based(label_list, pair_list, feature_name, MatchType.EXACT.value) for
                         feature_name in feature_names}
        partial_results = {feature_name: recall_json_based(label_list, pair_list, feature_name, MatchType.PARTIAL.value)
                           for
                           feature_name in feature_names}

        exact_results["R0 value"] = (recall_json_based(label_list, pair_list, "R0 value", MatchType.EXACT.value))
        partial_results["R0 value"] = (r0_partial_recall_json_based(label_list, pair_list))

        exact_results["overall"] = sum(exact_results.values()) / len(exact_results)
        partial_results["overall"] = sum(partial_results.values()) / len(partial_results)

        return exact_results, partial_results

    @staticmethod
    def all_precision_text_based(prediction_list, pair_list):

        exact_results = {
            feature_name: precision_text_based(prediction_list, pair_list, feature_name, MatchType.EXACT.value)
            for feature_name in feature_names}
        partial_results = {
            feature_name: precision_text_based(prediction_list, pair_list, feature_name, MatchType.PARTIAL.value)
            for feature_name in feature_names}

        exact_results["R0 value"] = (
            precision_text_based(prediction_list, pair_list, "R0 value", MatchType.EXACT.value))
        partial_results["R0 value"] = (r0_partial_precision_text_based(prediction_list, pair_list))

        exact_results["overall"] = sum(exact_results.values()) / len(exact_results)
        partial_results["overall"] = sum(partial_results.values()) / len(partial_results)

        return exact_results, partial_results

    @staticmethod
    def all_precision_json_based(prediction_list, pair_list):

        exact_results = {
            feature_name: precision_json_based(prediction_list, pair_list, feature_name, MatchType.EXACT.value)
            for feature_name in feature_names}
        partial_results = {
            feature_name: precision_json_based(prediction_list, pair_list, feature_name, MatchType.PARTIAL.value)
            for feature_name in feature_names}

        exact_results["R0 value"] = (
            precision_json_based(prediction_list, pair_list, "R0 value", MatchType.EXACT.value))
        partial_results["R0 value"] = (r0_partial_precision_json_based(prediction_list, pair_list))

        exact_results["overall"] = sum(exact_results.values()) / len(exact_results)
        partial_results["overall"] = sum(partial_results.values()) / len(partial_results)

        return exact_results, partial_results

    @staticmethod
    def evaluate_property_wise_text_based(label_list, prediction_list):

        evaluation_dict = compute_all_metrics_text_based(label_list, prediction_list)
        return {k: round(v * 100, 4) for k, v in flatten_result_dict(evaluation_dict).items()}

    @staticmethod
    def evaluate_property_wise_json_based(label_list, prediction_list):

        evaluation_dict = compute_all_metrics_json_based(label_list, prediction_list)
        return {k: round(v * 100, 4) for k, v in flatten_result_dict(evaluation_dict).items()}

    @staticmethod
    def evaluate_rouge(label_list, prediction_list):

        metric = evaluate.load("rouge")
        result = metric.compute(predictions=prediction_list, references=label_list, use_stemmer=True)
        return {k: round(v * 100, 4) for k, v in result.items()}
