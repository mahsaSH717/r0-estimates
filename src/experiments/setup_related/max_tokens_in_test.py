import json
import os
from datasets import DatasetDict
from transformers import T5Tokenizer
from tokenizers import AddedToken


data_path_text_based = "../../../data/processed/final_datasets/all_18_templates/text_based/orkg_brp_dataset"
data_path_json_based = "../../../data/processed/final_datasets/all_18_templates/json_based/orkg_brp_dataset"
result_path_test_text_based = "../../../experimental_results/setup_related/test/text_based"
result_path_dev_text_based = "../../../experimental_results/setup_related/dev/text_based"
result_path_train_text_based = "../../../experimental_results/setup_related/train/text_based"
result_path_test_json_based = "../../../experimental_results/setup_related/test/json_based"
result_path_dev_json_based = "../../../experimental_results/setup_related/dev/json_based"
result_path_train_json_based = "../../../experimental_results/setup_related/train/json_based"


tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
tokenizer.add_tokens(AddedToken("{", normalized=False))
tokenizer.add_tokens(AddedToken("}", normalized=False))
tokenizer.add_tokens(AddedToken("\n", normalized=False))

dataset_text_based = DatasetDict.load_from_disk(data_path_text_based)
dataset_json_based = DatasetDict.load_from_disk(data_path_json_based)


def save_json_to_path(path, json_data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as file:
        json.dump(json_data, file)


def tokenize_input(input_text):
    return tokenizer(input_text)["input_ids"]


def process_example(prompt_result_dict, response_result_dict, max_prompt_length, max_response_length,
                    example):
    template_name_param, template_number_param, cord_id, prompt, response = (
        example["template_name"],
        example["template_number"],
        example["cord_id"],
        example["prompt"],
        example["response"],
    )
    tokenized_prompt = tokenize_input(prompt)
    tokenized_response = tokenize_input(response)

    if len(tokenized_prompt) > 512:
        key = f"{cord_id}_{template_name_param}_{template_number_param}"
        prompt_result_dict[key] = {"prompt_len": len(tokenized_prompt), "prompt": prompt}
        max_prompt_length = max(max_prompt_length, len(tokenized_prompt))

    if len(tokenized_response) > 512:
        key = f"{cord_id}_{template_name_param}_{template_number_param}"
        response_result_dict[key] = {"response_len": len(tokenized_response), "response": response}
        max_response_length = max(max_response_length, len(tokenized_response))

    return prompt_result_dict, response_result_dict, max_prompt_length, max_response_length


def process_dataset(dataset_param, result_path_param):
    template_names = ['squad_v2', 'drop']
    template_numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

    prompt_result_dict = {}
    response_result_dict = {}
    max_prompt_length = 0
    max_response_length = 0

    test_dataset_group_by_template = [
        (template_name, template_number, filtered_dataset)
        for template_name in template_names
        for template_number in template_numbers
        if (filtered_dataset := dataset_param.filter(
            lambda item: item["template_name"] == template_name and item["template_number"] == template_number
        ))
        if len(filtered_dataset) > 0
    ]

    for template_name, template_number, filtered_dataset in test_dataset_group_by_template:
        for example in filtered_dataset:
            (
                prompt_result_dict,
                response_result_dict,
                max_prompt_length,
                max_response_length,
            ) = process_example(
                prompt_result_dict,
                response_result_dict,
                max_prompt_length,
                max_response_length,
                example,
            )

    summary = {
        "total_prompts": len(dataset_param["prompt"]),
        "total_responses": len(dataset_param["response"]),
        "prompt_dict_len": len(prompt_result_dict),
        "max_prompt_length": max_prompt_length,
        "response_dict_len": len(response_result_dict),
        "max_response_length": max_response_length,
    }

    save_json_to_path(result_path_param + "/prompt_max_tokens.json", prompt_result_dict)
    save_json_to_path(result_path_param + "/response_max_tokens.json", response_result_dict)
    save_json_to_path(result_path_param + "/summary.json", summary)


process_dataset(dataset_text_based['train'], result_path_train_text_based)
process_dataset(dataset_text_based['test'], result_path_test_text_based)
process_dataset(dataset_text_based['dev'], result_path_dev_text_based)
process_dataset(dataset_json_based['train'], result_path_train_json_based)
process_dataset(dataset_json_based['test'], result_path_test_json_based)
process_dataset(dataset_json_based['dev'], result_path_dev_json_based)
