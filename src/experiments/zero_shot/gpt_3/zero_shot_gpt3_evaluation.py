import os
import json
import pandas as pd
from datasets import DatasetDict
from src.util.evaluation_metrics_gpt3 import Metrics as em1
from src.util.evaluation_metrics import Metrics as em2


def load_test_dataset(path):
    dataset = DatasetDict.load_from_disk(path)
    return dataset['test']


def group_dataset_templates(test_set, template_names, template_numbers):
    return [(template_name, template_number, filtered_dataset) for template_name in template_names for template_number
            in template_numbers if (filtered_dataset := test_set.filter(
            lambda example: example["template_name"] == template_name and example[
                "template_number"] == template_number)) if len(filtered_dataset) > 0]


def load_gpt3_responses(filepath='', template_name='drop_1'):
    print('Loading the GPT-3 responses of the template: {}\n'.format(template_name))

    # Reading the excel file using pandas
    gpt3_data = pd.read_excel(filepath)

    # Returning only the responses column from the file
    return gpt3_data['Response'].tolist()


def evaluate_dataset(test_dataset_group_by_template, gpt3_predictions, frmt='text'):
    print('Evaluating the {} based dataset against the GPT3 Responses'.format(frmt))

    # A List to save the evaluation results for each template
    evaluation_results = []

    # Iterating over all templates
    for template_name, template_number, filtered_dataset in test_dataset_group_by_template:

        # Extracting all responses from the dataset
        test_set_response_list = [filtered_dataset[i]['response'] for i in range(filtered_dataset.num_rows)]

        # Extracting all gpt3 responses for specified template
        gpt3_template_predictions = gpt3_predictions['{}_{}'.format(template_name, template_number)]

        # Evaluating the responses
        result = em1.evaluate_rouge(test_set_response_list, gpt3_template_predictions)

        if template_name == 'drop':
            result["general_accuracy"] = round(
                em1.general_accuracy_text_based(test_set_response_list, gpt3_template_predictions) * 100,
                4) if frmt == 'text' else round(
                em1.general_accuracy_json_based(test_set_response_list, gpt3_template_predictions) * 100, 4)
        else:
            result["general_accuracy"] = round(
                em2.general_accuracy_text_based(test_set_response_list, gpt3_template_predictions) * 100,
                4) if frmt == 'text' else round(
                em2.general_accuracy_json_based(test_set_response_list, gpt3_template_predictions) * 100, 4)

        evaluation_results.append(
            {'template_name': template_name, 'template_number': template_number, 'metrics': result})

    print('Evaluation done!\n')

    # Returning the evaluation results
    return evaluation_results


def save_json(evaluation_results, directory='../../../../experimental_results/zero_shot/gpt_3/',
              filename='zero_shot_result_gpt3_on_test.json'):
    # Checking the directory and the filename
    file_path = directory + filename
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Saving the data as a json dump
    with open(file_path, 'w') as file:
        # Use json to dump the list as JSON into the file
        json.dump(evaluation_results, file)

    print('Evaluation Results saved at location: {}\n'.format(file_path))


if __name__ == "__main__":

    template_names = ['squad_v2', 'drop']
    template_numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

    test_set_text_based = load_test_dataset(
        '../../../../data/processed/final_datasets/all_18_templates/text_based/orkg_brp_dataset')
    test_set_json_based = load_test_dataset(
        '../../../../data/processed/final_datasets/all_18_templates/json_based/orkg_brp_dataset')

    test_dataset_text_based_group_by_template = group_dataset_templates(test_set_text_based, template_names,
                                                                        template_numbers)
    test_dataset_json_based_group_by_template = group_dataset_templates(test_set_json_based, template_names,
                                                                        template_numbers)

    # Dictionary to save gpt3 responses per template
    gpt3_predictions = {}

    for temp_name in template_names:
        for temp_num in template_numbers:
            filename = 'gpt3_{}_{}_results.xlsx'.format(temp_name, temp_num)

            if not os.path.exists('../../../../experimental_results/zero_shot/gpt_3/' + filename):
                continue

            gpt3_responses = load_gpt3_responses('../../../../experimental_results/zero_shot/gpt_3/' + filename,
                                                 template_name='{}_{}'.format(temp_name, temp_num))

            # Saving the gpt3 response for each template
            gpt3_predictions['{}_{}'.format(temp_name, temp_num)] = gpt3_responses

    evaluation_results_text_based = evaluate_dataset(test_dataset_text_based_group_by_template, gpt3_predictions,
                                                     frmt='text')
    evaluation_results_json_based = evaluate_dataset(test_dataset_json_based_group_by_template, gpt3_predictions,
                                                     frmt='json')

    save_json(evaluation_results_text_based, filename='zero_shot_result_text_based_gpt3_on_test.json')
    save_json(evaluation_results_json_based, filename='zero_shot_result_json_based_gpt3_on_test.json')
