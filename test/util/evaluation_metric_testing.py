import re
import unittest
import random
from evaluation_metrics import Metrics
from datasets import DatasetDict
import src.util.evaluation_metrics as em

class DataFormatter:

    def __init__(self):
        self.dataset = DatasetDict.load_from_disk("../../data/processed/hg_datasets/drop_1/text_based/orkg_brp_dataset")
        self.dataset = self.dataset.shuffle(seed=42)

    def create_dummy_summary(self, text, pattern, sep='\n', total_words=100):
        # Extracting the abstract and taking some words to create a dummy summary
        match = re.search(pattern, text)
        if match:
            text = match.group(1)
        return ' '.join(text.split()[:total_words])

    def format_data_rouge(self):
        # Defining the pattern to extract the abstract
        summary_pattern = r"BACKGROUND:\s*(.*)\n"

        # Taking first 100 words of the first two paper abstracts
        self.summary1 = self.create_dummy_summary(self.dataset['train'][0]['prompt'], summary_pattern)
        self.summary2 = self.create_dummy_summary(self.dataset['train'][1]['prompt'], summary_pattern)

    def update_tuple_value(self, grouped_texts, update_value, tuple_index, total_random_samples=5):
        # Randomly sampling some grouped texts
        imputed_data = {i: grouped_texts[i] for i in random.sample(range(len(grouped_texts)), total_random_samples)}

        # Updating the value of a field specified.
        updated_lst = []
        for key, value in imputed_data.items():
            for v in value:
                v = list(v)
                v[tuple_index] = update_value
                updated_lst.append(tuple(v))
            imputed_data[key] = updated_lst
        return imputed_data

    def format_tuple_to_string(self, lst_of_tup):
        return '|'.join('\n'.join(tup) for tup in lst_of_tup)

    def format_response_data(self):
        # First picking 100 random instances from the dataset
        self.random_instances = random.sample([x for x in self.dataset['train'][:]['response'] if x != 'unanswerable'],
                                              100)

        # Using regular expression to group the information
        grouped_texts = []
        regex = r"(disease name:\s[^\n]*)\n(location:\s[^\n]*)\n(date:\s[^\n]*)\n(R0 value:\s[^\n]*)\n(%CI values:\s[^\n]*)\n(method:\s[^\n]*)"
        for instance in self.random_instances:
            grouped_texts.append(re.findall(regex, instance, re.DOTALL))

        # Changing the structure of the first 20 responses
        self.new_format_response = []
        for gt in grouped_texts[:20]:
            self.new_format_response.append('\n'.join([
                                                          f"{contribution[0]}, {contribution[1]}, {contribution[3]}, {contribution[2]}, {contribution[5]}, {contribution[4]}"
                                                          for contribution in gt]))

        # Taking 5 random instances from above samples and changing Disease name
        self.disease_imputed = self.update_tuple_value(grouped_texts, 'disease name: Some Disease name', 0)

        # Taking 5 random instances from above samples and changing Location name
        self.location_imputed = self.update_tuple_value(grouped_texts, 'location: Some location', 1)

        # Taking 5 random instances from above samples and changing Date
        self.date_imputed = self.update_tuple_value(grouped_texts, 'date: Some date', 2)

        # Taking 5 random instances from above samples and changing R0 value
        self.r0_imputed = self.update_tuple_value(grouped_texts, 'R0 value: Some R0 value', 3)

        # Taking 5 random instances from above samples and changing CI values
        self.ci_imputed = self.update_tuple_value(grouped_texts, '%CI values: Some CI values', 4)

        # Taking 5 random instances from above samples and changing R0 value
        self.method_imputed = self.update_tuple_value(grouped_texts, 'method: Some method', 5)

    def main(self):
        # Formatting data to be used in testing ROUGE
        self.format_data_rouge()

        # Formatting responses to be used in testing all metrics
        self.format_response_data()
        return self


class EvaluationMetricsTest(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.data_formatter = DataFormatter().main()

    def test_rouge(self):
        # Testing ROUGE for two identical examples
        summary = ["The quick brown fox jumped over the lazy dog"]
        reference = ["The quick brown fox jumped over the lazy dog"]
        assert all(val == 100 for val in Metrics.evaluate_rouge(summary, reference).values())

        # Generating summaries from the two abstracts to check similarity
        assert all(val > 0 for val in
                   Metrics.evaluate_rouge([self.data_formatter.summary1], [self.data_formatter.summary1]).values())

        # Taking some responses from the dataset and modifying their disease value to check ROUGE score. The test passes if all the scores are less than 100
        updated_instances = [self.data_formatter.format_tuple_to_string(self.data_formatter.disease_imputed[i])
                             if i in self.data_formatter.disease_imputed.keys() else inst for i, inst in
                             enumerate(self.data_formatter.random_instances)]
        assert all(val < 100 for val in
                   Metrics.evaluate_rouge(self.data_formatter.random_instances, updated_instances).values())

        # Taking some responses from the dataset and modifying their location value to check ROUGE score. The test passes if all the scores are less than 100
        updated_instances = [self.data_formatter.format_tuple_to_string(self.data_formatter.location_imputed[i])
                             if i in self.data_formatter.location_imputed.keys() else inst for i, inst in
                             enumerate(self.data_formatter.random_instances)]
        assert all(val < 100 for val in
                   Metrics.evaluate_rouge(self.data_formatter.random_instances, updated_instances).values())

        # Taking some responses from the dataset and modifying their Date value to check ROUGE score. The test passes if all the scores are less than 100
        updated_instances = [self.data_formatter.format_tuple_to_string(self.data_formatter.date_imputed[i])
                             if i in self.data_formatter.date_imputed.keys() else inst for i, inst in
                             enumerate(self.data_formatter.random_instances)]
        assert all(val < 100 for val in
                   Metrics.evaluate_rouge(self.data_formatter.random_instances, updated_instances).values())

        # Taking some responses from the dataset and modifying their R0 value to check ROUGE score. The test passes if all the scores are less than 100
        updated_instances = [self.data_formatter.format_tuple_to_string(self.data_formatter.r0_imputed[i])
                             if i in self.data_formatter.r0_imputed.keys() else inst for i, inst in
                             enumerate(self.data_formatter.random_instances)]
        assert all(val < 100 for val in
                   Metrics.evaluate_rouge(self.data_formatter.random_instances, updated_instances).values())

        # Taking some responses from the dataset and modifying their CI value to check ROUGE score. The test passes if all the scores are less than 100
        updated_instances = [self.data_formatter.format_tuple_to_string(self.data_formatter.ci_imputed[i])
                             if i in self.data_formatter.ci_imputed.keys() else inst for i, inst in
                             enumerate(self.data_formatter.random_instances)]
        assert all(val < 100 for val in
                   Metrics.evaluate_rouge(self.data_formatter.random_instances, updated_instances).values())

        # Taking some responses from the dataset and modifying their Method value to check ROUGE score. The test passes if all the scores are less than 100
        updated_instances = [self.data_formatter.format_tuple_to_string(self.data_formatter.method_imputed[i])
                             if i in self.data_formatter.method_imputed.keys() else inst for i, inst in
                             enumerate(self.data_formatter.random_instances)]
        assert all(val < 100 for val in
                   Metrics.evaluate_rouge(self.data_formatter.random_instances, updated_instances).values())

    def test_recall(self):
        # Replacing the first 20 response with the restructured responses
        predictions_restructured = self.data_formatter.random_instances.copy()
        predictions_restructured[:20] = self.data_formatter.new_format_response.copy()

        # Since the responses are restructured, the test will be passed if all the exact and partial recall values are less than 1
        exact_recall, partial_recall = Metrics.all_recall_text_based(self.data_formatter.random_instances,
                                                                     em.make_list_of_pairs(
                                                                         self.data_formatter.random_instances,
                                                                         predictions_restructured))
        assert all(val < 1 for val in exact_recall.values()) and all(val < 1 for val in partial_recall.values())

        # Taking some responses from the dataset and modifying their disease value to check recall score. The test will pass if the exact and partial recall values of disease is less than 1
        updated_instances = [self.data_formatter.format_tuple_to_string(self.data_formatter.disease_imputed[i])
                             if i in self.data_formatter.disease_imputed.keys() else inst for i, inst in
                             enumerate(self.data_formatter.random_instances)]

        assert all(val['disease name'] < 1 for val in
                   Metrics.all_recall_text_based(self.data_formatter.random_instances,
                                                 em.make_list_of_pairs(self.data_formatter.random_instances,
                                                                       updated_instances)))

        # Taking some responses from the dataset and modifying their location value to check recall score. The test will pass if the exact and partial recall values of location is less than 1
        updated_instances = [self.data_formatter.format_tuple_to_string(self.data_formatter.location_imputed[i])
                             if i in self.data_formatter.location_imputed.keys() else inst for i, inst in
                             enumerate(self.data_formatter.random_instances)]
        assert all(val['location'] < 1 for val in Metrics.all_recall_text_based(self.data_formatter.random_instances,
                                                                                em.make_list_of_pairs(
                                                                                    self.data_formatter.random_instances,
                                                                                    updated_instances)))

        # Taking some responses from the dataset and modifying their date value to check recall score. The test will pass if the exact and partial recall values of date is less than 1
        updated_instances = [self.data_formatter.format_tuple_to_string(self.data_formatter.date_imputed[i])
                             if i in self.data_formatter.date_imputed.keys() else inst for i, inst in
                             enumerate(self.data_formatter.random_instances)]
        assert all(val['date'] < 1 for val in Metrics.all_recall_text_based(self.data_formatter.random_instances,
                                                                            em.make_list_of_pairs(
                                                                                self.data_formatter.random_instances,
                                                                                updated_instances)))

        # Taking some responses from the dataset and modifying their CI  value to check recall score. The test will pass if the exact and partial recall values of CI is less than 1
        updated_instances = [self.data_formatter.format_tuple_to_string(self.data_formatter.ci_imputed[i])
                             if i in self.data_formatter.ci_imputed.keys() else inst for i, inst in
                             enumerate(self.data_formatter.random_instances)]
        assert all(val['%CI values'] < 1 for val in Metrics.all_recall_text_based(self.data_formatter.random_instances,
                                                                                  em.make_list_of_pairs(
                                                                                      self.data_formatter.random_instances,
                                                                                      updated_instances)))

        # Taking some responses from the dataset and modifying their method value to check recall score. The test will pass if the exact and partial recall values of method is less than 1
        updated_instances = [self.data_formatter.format_tuple_to_string(self.data_formatter.method_imputed[i])
                             if i in self.data_formatter.method_imputed.keys() else inst for i, inst in
                             enumerate(self.data_formatter.random_instances)]
        assert all(val['method'] < 1 for val in Metrics.all_recall_text_based(self.data_formatter.random_instances,
                                                                              em.make_list_of_pairs(
                                                                                  self.data_formatter.random_instances,
                                                                                  updated_instances)))

        # Taking some responses from the dataset and modifying their R0 value to check recall score. The test will pass if the exact and partial recall values of R0 is less than 1
        updated_instances = [self.data_formatter.format_tuple_to_string(self.data_formatter.r0_imputed[i])
                             if i in self.data_formatter.r0_imputed.keys() else inst for i, inst in
                             enumerate(self.data_formatter.random_instances)]
        assert all(val['R0 value'] < 1 for val in Metrics.all_recall_text_based(self.data_formatter.random_instances,
                                                                                em.make_list_of_pairs(
                                                                                    self.data_formatter.random_instances,
                                                                                    updated_instances)))

    def test_precision(self):
        # Replacing the first 20 response with the restructured responses
        predictions_restructured = self.data_formatter.random_instances.copy()
        predictions_restructured[:20] = self.data_formatter.new_format_response.copy()

        # Since the responses are restructured, the test will be passed if all the exact and partial precision values are less than 1
        exact_precision, partial_precision = Metrics.all_precision_text(predictions_restructured, em.make_list_of_pairs(
            self.data_formatter.random_instances, predictions_restructured))
        assert all(val < 1 for val in exact_precision.values()) and all(val < 1 for val in partial_precision.values())

        # Taking some responses from the dataset and modifying their disease value to check precision score. The test will pass if the exact and partial precsison values of disease is less than 1
        updated_instances = [self.data_formatter.format_tuple_to_string(self.data_formatter.disease_imputed[i])
                             if i in self.data_formatter.disease_imputed.keys() else inst for i, inst in
                             enumerate(self.data_formatter.random_instances)]
        assert all(val['disease name'] < 1 for val in Metrics.all_precision_text(updated_instances,
                                                                                 em.make_list_of_pairs(
                                                                                     self.data_formatter.random_instances,
                                                                                     updated_instances)))

        # Taking some responses from the dataset and modifying their location value to check precision score. The test will pass if the exact and partial precision values of location is less than 1
        updated_instances = [self.data_formatter.format_tuple_to_string(self.data_formatter.location_imputed[i])
                             if i in self.data_formatter.location_imputed.keys() else inst for i, inst in
                             enumerate(self.data_formatter.random_instances)]
        assert all(val['location'] < 1 for val in Metrics.all_precision_text(updated_instances, em.make_list_of_pairs(
            self.data_formatter.random_instances, updated_instances)))

        # Taking some responses from the dataset and modifying their date value to check precision score. The test will pass if the exact and partial precision values of date is less than 1
        updated_instances = [self.data_formatter.format_tuple_to_string(self.data_formatter.date_imputed[i])
                             if i in self.data_formatter.date_imputed.keys() else inst for i, inst in
                             enumerate(self.data_formatter.random_instances)]
        assert all(val['date'] < 1 for val in Metrics.all_precision_text(updated_instances, em.make_list_of_pairs(
            self.data_formatter.random_instances, updated_instances)))

        # Taking some responses from the dataset and modifying their CI  value to check precision score. The test will pass if the exact and partial precision values of CI is less than 1
        updated_instances = [self.data_formatter.format_tuple_to_string(self.data_formatter.ci_imputed[i])
                             if i in self.data_formatter.ci_imputed.keys() else inst for i, inst in
                             enumerate(self.data_formatter.random_instances)]
        assert all(val['%CI values'] < 1 for val in Metrics.all_precision_text(updated_instances, em.make_list_of_pairs(
            self.data_formatter.random_instances, updated_instances)))

        # Taking some responses from the dataset and modifying their method value to check precision score. The test will pass if the exact and partial precision values of method is less than 1
        updated_instances = [self.data_formatter.format_tuple_to_string(self.data_formatter.method_imputed[i])
                             if i in self.data_formatter.method_imputed.keys() else inst for i, inst in
                             enumerate(self.data_formatter.random_instances)]
        assert all(val['method'] < 1 for val in Metrics.all_precision_text(updated_instances, em.make_list_of_pairs(
            self.data_formatter.random_instances, updated_instances)))

        # Taking some responses from the dataset and modifying their R0 value to check precision score. The test will pass if the exact and partial precision values of R0 is less than 1
        updated_instances = [self.data_formatter.format_tuple_to_string(self.data_formatter.r0_imputed[i])
                             if i in self.data_formatter.r0_imputed.keys() else inst for i, inst in
                             enumerate(self.data_formatter.random_instances)]
        assert all(val['R0 value'] < 1 for val in Metrics.all_precision_text(updated_instances, em.make_list_of_pairs(
            self.data_formatter.random_instances, updated_instances)))

    def test_f1_score(self):
        # Replacing the first 20 response with the restructured responses
        predictions_restructured = self.data_formatter.random_instances.copy()
        predictions_restructured[:20] = self.data_formatter.new_format_response.copy()

        # Since the responses are restructured, the test will be passed if all the exact and partial f1 score values are less than 1
        exact_f1s = em.compute_all_metrics_text_based(self.data_formatter.random_instances, predictions_restructured)[
            'exact_f1s']
        partial_f1s = em.compute_all_metrics_text_based(self.data_formatter.random_instances, predictions_restructured)[
            'partial_f1s']
        assert all(val < 1 for val in exact_f1s.values()) and all(val < 1 for val in partial_f1s.values())

        # Taking some responses from the dataset and modifying their disease value to check f1 score. The test will pass if the exact and partial f1 values of disease is less than 1
        updated_instances = [self.data_formatter.format_tuple_to_string(self.data_formatter.disease_imputed[i])
                             if i in self.data_formatter.disease_imputed.keys() else inst for i, inst in
                             enumerate(self.data_formatter.random_instances)]
        assert all(val['disease name'] < 1 for val in
                   [em.compute_all_metrics_text_based(self.data_formatter.random_instances, updated_instances)[key] for
                    key in ['exact_f1s', 'partial_f1s']])

        # Taking some responses from the dataset and modifying their location value to check f1 score. The test will pass if the exact and partial f1 values of location is less than 1
        updated_instances = [self.data_formatter.format_tuple_to_string(self.data_formatter.location_imputed[i])
                             if i in self.data_formatter.location_imputed.keys() else inst for i, inst in
                             enumerate(self.data_formatter.random_instances)]
        assert all(val['location'] < 1 for val in
                   [em.compute_all_metrics_text_based(self.data_formatter.random_instances, updated_instances)[key] for
                    key in ['exact_f1s', 'partial_f1s']])

        # Taking some responses from the dataset and modifying their date value to check f1 score. The test will pass if the exact and partial f1 values of date is less than 1
        updated_instances = [self.data_formatter.format_tuple_to_string(self.data_formatter.date_imputed[i])
                             if i in self.data_formatter.date_imputed.keys() else inst for i, inst in
                             enumerate(self.data_formatter.random_instances)]
        assert all(val['date'] < 1 for val in
                   [em.compute_all_metrics_text_based(self.data_formatter.random_instances, updated_instances)[key] for
                    key in ['exact_f1s', 'partial_f1s']])

        # Taking some responses from the dataset and modifying their CI  value to check f1 score. The test will pass if the exact and partial f1 values of CI is less than 1
        updated_instances = [self.data_formatter.format_tuple_to_string(self.data_formatter.ci_imputed[i])
                             if i in self.data_formatter.ci_imputed.keys() else inst for i, inst in
                             enumerate(self.data_formatter.random_instances)]
        assert all(val['%CI values'] < 1 for val in
                   [em.compute_all_metrics_text_based(self.data_formatter.random_instances, updated_instances)[key] for
                    key in ['exact_f1s', 'partial_f1s']])

        # Taking some responses from the dataset and modifying their method value to check f1 score. The test will pass if the exact and partial f1 values of method is less than 1
        updated_instances = [self.data_formatter.format_tuple_to_string(self.data_formatter.method_imputed[i])
                             if i in self.data_formatter.method_imputed.keys() else inst for i, inst in
                             enumerate(self.data_formatter.random_instances)]
        assert all(val['method'] < 1 for val in
                   [em.compute_all_metrics_text_based(self.data_formatter.random_instances, updated_instances)[key] for
                    key in ['exact_f1s', 'partial_f1s']])

        # Taking some responses from the dataset and modifying their R0 value to check f1 score. The test will pass if the exact and partial f1 values of R0 is less than 1
        updated_instances = [self.data_formatter.format_tuple_to_string(self.data_formatter.r0_imputed[i])
                             if i in self.data_formatter.r0_imputed.keys() else inst for i, inst in
                             enumerate(self.data_formatter.random_instances)]
        assert all(val['R0 value'] < 1 for val in
                   [em.compute_all_metrics_text_based(self.data_formatter.random_instances, updated_instances)[key] for
                    key in ['exact_f1s', 'partial_f1s']])


if __name__ == '__main__':
    unittest.main()
