import os
import json
import time
import openai
import openpyxl
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


def request_openai(prompt, model='text-davinci-003', max_token=1330, temperature=0):
    recieved_response = False
    while not recieved_response:
        try:
            response = openai.Completion.create(model=model, prompt=prompt, max_tokens=max_token,
                                                temperature=temperature)
            recieved_response = True
        except openai.error.RateLimitError as rl:
            print('Exception occurred: {}'.format(rl))
            print('Sleeping for 1 minute')
            time.sleep(60)
        except Exception as e:
            print('Exception occured, trying again')
    return response


def gather_gpt3_responses(test_dataset_group_by_template,
                          save_filepath='../../../../experimental_results/zero_shot/gpt_3'):
    print('Gathering the GPT3 Responses')

    # Creating a new workbook and getting the active sheet
    combined_workbook = openpyxl.Workbook()
    cw_sheet = combined_workbook.active

    # Appending the First row Headers
    cw_sheet.append(['Template', 'Prompt', 'Response'])

    # Dictionary to save gpt3 responses per template
    gpt3_predictions = {}

    # Iterating over all templates
    for template_name, template_number, filtered_dataset in test_dataset_group_by_template:

        print('\nStarting the template: {}'.format('{}_{}'.format(template_name, template_number)))

        # Creating a new workbook for this specific template and getting the active sheet
        template_workbook = openpyxl.Workbook()
        t_sheet = template_workbook.active

        # Appending the First row Headers
        t_sheet.append(['Template', 'Prompt', 'Response'])

        # Extracting the prompt from each observation
        test_set_prompt_list = [filtered_dataset[i]['prompt'] for i in range(filtered_dataset.num_rows)]

        print('Total Prompt: {}'.format(len(test_set_prompt_list)))

        # Iterating over all prompts and getting gpt3 response
        gpt3_template_predictions = []
        for index, prompt in enumerate(test_set_prompt_list):
            gpt3_response = request_openai(prompt)
            gpt3_template_predictions.append(gpt3_response['choices'][0]['text'])

            # Appending the outputs from GPT3 to the excel file
            cw_sheet.append(
                ['{}_{}'.format(template_name, template_number), prompt, gpt3_response['choices'][0]['text']])
            t_sheet.append(
                ['{}_{}'.format(template_name, template_number), prompt, gpt3_response['choices'][0]['text']])

            # Waiting 5 seconds after every request
            time.sleep(5)

            if (index % 100) == 0:
                print('Total Prompt Completed: {}'.format(index + 1))

        # Saving the gpt3 response for each template
        gpt3_predictions['{}_{}'.format(template_name, template_number)] = gpt3_template_predictions

        print('Saving the {}_{} Results in the excel file'.format(template_name, template_number))

        # Applying some styling to the excel sheet
        t_sheet.column_dimensions['A'].width = 25
        t_sheet.column_dimensions['B'].width = 100
        t_sheet.column_dimensions['C'].width = 100

        # Saving the excel file
        template_workbook.save(
            filename=save_filepath + '/gpt3_{}_{}_results.xlsx'.format(template_name, template_number))

        # Waiting for some seconds according to the API RPM
        print('Template: {} is finished, waiting for 10 seconds'.format('{}_{}'.format(template_name, template_number)))
        time.sleep(10)

    # Applying some styling to the excel sheet
    cw_sheet.column_dimensions['A'].width = 25
    cw_sheet.column_dimensions['B'].width = 100
    cw_sheet.column_dimensions['C'].width = 100

    # Saving the excel file
    combined_workbook.save(filename=save_filepath + '/gpt3_combined_results.xlsx')

    print('GPT3 Responses saved to the location: {}\n'.format(save_filepath))

    # Returning all gpt3 responses
    return gpt3_predictions


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
              filename='/zero_shot_result_gpt3_on_test.json'):
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

    openai.organization = ''
    openai.api_key = ''

    gpt3_predictions = gather_gpt3_responses(test_dataset_text_based_group_by_template)

    evaluation_results_text_based = evaluate_dataset(test_dataset_text_based_group_by_template, gpt3_predictions,
                                                     frmt='text')
    evaluation_results_json_based = evaluate_dataset(test_dataset_json_based_group_by_template, gpt3_predictions,
                                                     frmt='json')

    save_json(evaluation_results_text_based, filename='/zero_shot_result_text_based_gpt3_on_test.json')
    save_json(evaluation_results_json_based, filename='/zero_shot_result_json_based_gpt3_on_test.json')
