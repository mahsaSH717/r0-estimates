import json
import os
import random

import numpy as np
import torch
from datasets import DatasetDict
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM
from tokenizers import AddedToken

from src.util.evaluation_metrics import Metrics

seed = 42
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

model_is_from = "drop_5"
desired_checkpoint = "checkpoint-1623"
path_prefix = "../../../.."
tokenizer_path = f"{path_prefix}/models/text_based/{model_is_from}/tokenizer"
model_path = f"{path_prefix}/models/text_based/{model_is_from}/{desired_checkpoint}"
dataset_path = f"{path_prefix}/data/processed/final_datasets/{model_is_from}/text_based/orkg_brp_dataset"
output_dir = f"{path_prefix}/general_logs/text_based/{model_is_from}"
result_path = f"{path_prefix}/experimental_results/flan_t5_finetuning/text_based/{model_is_from}/{desired_checkpoint}"

os.makedirs(os.path.dirname(output_dir), exist_ok=True)
os.makedirs(os.path.dirname(result_path), exist_ok=True)

template_names = ['squad_v2', 'drop']
template_numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
batch_size = 16
max_target_length = 1024
max_source_length = 512

dataset_main = DatasetDict.load_from_disk(dataset_path)
test_dataset = dataset_main['test']

tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
tokenizer.add_tokens(AddedToken("\n", normalized=False))

model = AutoModelForSeq2SeqLM.from_pretrained(model_path, device_map="auto")

test_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_eval_batch_size=batch_size,
    predict_with_generate=True,
    generation_max_length=max_target_length,
    do_train=False,
    do_predict=True,
    dataloader_drop_last=False
)

trainer = Seq2SeqTrainer(
    model=model,
    args=test_args
)


def prompt_tokenize_function(sample):
    # tokenize inputs
    return tokenizer(sample["prompt"], max_length=max_source_length, padding="max_length", truncation=True,
                     return_tensors="pt")


def get_decoded_response_list(test_set_response_list):
    tokenized_responses = tokenizer(test_set_response_list, max_length=max_target_length, padding="max_length",
                                    return_tensors="pt")
    return tokenizer.batch_decode(tokenized_responses["input_ids"], skip_special_tokens=True)


def compute_metrics(preds, labels):
    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)  # type: ignore
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    print(len(decoded_preds))
    print(len(labels))
    result = Metrics.evaluate_property_wise_text_based(label_list=labels, prediction_list=decoded_preds)
    result.update(Metrics.evaluate_rouge(label_list=labels, prediction_list=decoded_preds))
    return result


def prediction(template_name, template_number, dataset):
    test_set_response_list = [dataset[i]['response'] for i in range(dataset.num_rows)]
    decoded_test_set_response_list = get_decoded_response_list(test_set_response_list)
    prediction_output = trainer.predict(dataset)
    predictions = prediction_output.predictions
    print("predictions_len: ", len(predictions))
    print("labels_len: ", len(decoded_test_set_response_list))
    result = compute_metrics(preds=predictions, labels=decoded_test_set_response_list)
    return template_name, template_number, result


def save_json_results(path, json_results):
    if not os.path.exists(path):
        os.makedirs(path)

    with open(f"{path}/predictions_results.json", 'w') as file:
        json.dump(json_results, file)


tokenized_test_dataset = test_dataset.map(prompt_tokenize_function, batched=True)

test_dataset_group_by_template = [(template_name, template_number, filtered_dataset) for template_name in template_names
                                  for template_number in template_numbers if
                                  (filtered_dataset := tokenized_test_dataset.filter(
                                      lambda example: example["template_name"] == template_name and example[
                                          "template_number"] == template_number)) if len(filtered_dataset) > 0]

pred_results = [prediction(template_name, template_number, dataset) for template_name, template_number, dataset in
                test_dataset_group_by_template]

predictions_results_json = [{'template_name': t[0], 'template_number': t[1], 'metrics': t[2]} for t in pred_results]

save_json_results(result_path, predictions_results_json)
