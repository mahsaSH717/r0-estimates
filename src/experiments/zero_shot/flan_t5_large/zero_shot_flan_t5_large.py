import json
import os

import torch
from datasets import DatasetDict
from tokenizers import AddedToken
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from transformers import T5Tokenizer, T5ForConditionalGeneration, GenerationConfig

from src.util.evaluation_metrics import Metrics

seed = 42
torch.cuda.manual_seed_all(seed)
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
tokenizer.add_tokens(AddedToken("\n", normalized=False))
tokenizer.add_tokens(AddedToken("}", normalized=False))
tokenizer.add_tokens(AddedToken("{", normalized=False))

model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large", device_map="auto")
generation_config = GenerationConfig.from_pretrained("google/flan-t5-large")
generation_config.max_new_tokens = 1330
batch_size = 16

path_prefix = "../../../.."

text_dataset_path = f"{path_prefix}/data/processed/final_datasets/all_18_templates/text_based/orkg_brp_dataset"
json_dataset_path = f"{path_prefix}/data/processed/final_datasets/all_18_templates/json_based/orkg_brp_dataset"
text_result_path = f"{path_prefix}/experimental_results/zero_shot/flan_t5_large/flan_t5_large_zero_shot_text_based_results.json"
json_result_path = f"{path_prefix}/experimental_results/zero_shot/flan_t5_large/flan_t5_large_zero_shot_json_based_results.json"
prediction_result_path = f"{path_prefix}/experimental_results/zero_shot/flan_t5_large/prediction_results"
template_names = ['squad_v2', 'drop']
template_numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']


def save_json_to_path(path, json_data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as file:
        json.dump(json_data, file)


def save_list_to_file(strings, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as file:
        for string in strings:
            file.write(string + '\n')


def load_datasets(data_path):
    dataset = DatasetDict.load_from_disk(data_path)
    return dataset['test']


def generate_predictions(test_set_prompt_list):
    decoded_preds = []
    input_ids_val = tokenizer(test_set_prompt_list, max_length=512, padding="max_length", truncation=True,
                              return_tensors="pt").input_ids.to("cuda")

    dataset_val = TensorDataset(input_ids_val)
    dataloader_val = DataLoader(dataset_val, sampler=SequentialSampler(dataset_val), batch_size=batch_size)
    for batch in dataloader_val:
        output = model.generate(batch[0], generation_config)
        decoded_batch = tokenizer.batch_decode(output, skip_special_tokens=True)
        decoded_preds.extend(decoded_batch)
    return decoded_preds


def evaluate_dataset(dataset_text_based, dataset_json_based):
    evaluation_results_text_based = []
    evaluation_results_json_based = []
    dataset_text_based_group_by_template = group_by_datasets(dataset_text_based)
    dataset_json_based_group_by_template = group_by_datasets(dataset_json_based)

    test_set_response_list_text_based = filter_dataset_column_to_list(dataset_text_based_group_by_template[0][2],
                                                                      'response')
    test_set_response_list_json_based = filter_dataset_column_to_list(dataset_json_based_group_by_template[0][2],
                                                                      'response')

    for template_name, template_number, filtered_dataset in dataset_text_based_group_by_template:
        test_set_prompt_list = filter_dataset_column_to_list(filtered_dataset, 'prompt')
        template_decoded_preds = generate_predictions(test_set_prompt_list)
        save_list_to_file(template_decoded_preds,
                          f"{prediction_result_path}/{template_name}_{template_number}_flan_t5_large_zero_shot_prediction_results.txt")
        text_evaluation = get_evaluation_results_text_based(template_decoded_preds, test_set_response_list_text_based)
        json_evaluation = get_evaluation_results_json_based(template_decoded_preds, test_set_response_list_json_based)
        fill_results(evaluation_results_text_based, template_name, template_number, text_evaluation)
        fill_results(evaluation_results_json_based, template_name, template_number, json_evaluation)

    return evaluation_results_text_based, evaluation_results_json_based


def group_by_datasets(dataset):
    return [(template_name, template_number, filtered_dataset) for template_name in
            template_names for template_number in template_numbers if (
                filtered_dataset := dataset.filter(lambda example: example["template_name"] == template_name and
                                                                   example["template_number"] == template_number)) if
            len(filtered_dataset) > 0]


def filter_dataset_column_to_list(filtered_dataset_text_based, column_name):
    test_set_response_list_text_based = [filtered_dataset_text_based[i][column_name] for i in
                                         range(filtered_dataset_text_based.num_rows)]
    return test_set_response_list_text_based


def fill_results(evaluation_results_text_based, template_name, template_number, evaluation_results):
    evaluation_results_text_based.append({
        'template_name': template_name,
        'template_number': template_number,
        'metrics': evaluation_results
    })


def get_evaluation_results_text_based(template_decoded_preds, test_set_response_list):
    result = Metrics.evaluate_rouge(test_set_response_list, template_decoded_preds)
    result["general_accuracy"] = round(
        Metrics.general_accuracy_text_based(test_set_response_list, template_decoded_preds) * 100, 4)
    return result


def get_evaluation_results_json_based(template_decoded_preds, test_set_response_list):
    result = Metrics.evaluate_rouge(test_set_response_list, template_decoded_preds)
    result["general_accuracy"] = round(
        Metrics.general_accuracy_json_based(test_set_response_list, template_decoded_preds) * 100, 4)
    return result


def evaluate_and_save_results():
    dataset_text_based = load_datasets(text_dataset_path)
    dataset_json_based = load_datasets(json_dataset_path)
    evaluation_results_text_based, evaluation_results_json_based = evaluate_dataset(dataset_text_based,
                                                                                    dataset_json_based)
    save_json_to_path(text_result_path, evaluation_results_text_based)
    save_json_to_path(json_result_path, evaluation_results_json_based)


evaluate_and_save_results()
