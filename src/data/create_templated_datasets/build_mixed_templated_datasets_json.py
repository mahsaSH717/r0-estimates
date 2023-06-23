import json
import math
import os
import random

import pandas as pd
from datasets import Dataset, DatasetDict

best_patterns_list = [

    ("drop", 3, "{context}\n\n{question}", "{answer}"),
    ("drop", 6, "{context}\n\nBased on the above article, answer a question. {question}", "{answer}"),
    ("squad_v2", 4, "{context}\n{question} (If the question is unanswerable, say \"unanswerable\")", "{answer}"),
    ("squad_v2", 6, "{context}\nIf it is possible to answer this question, answer it for me (else, reply \"unanswerable\"): {question}", "{answer}"),
    ("squad_v2", 8, "Read this: {context}\n\n{question}\nWhat is the answer? (If it cannot be answered, return \"unanswerable\")", "{answer}"),
    ("squad_v2", 9, "Read this: {context}\nNow answer this question, if there is an answer (If it cannot be answered, return \"unanswerable\"): {question}", "{answer}"),
    ("squad_v2", 10, "{context}\nIs there an answer to this question (If it cannot be answered, say \"unanswerable\"): {question}", "{answer}")

]

question = '''What are the values for the following properties of the basic reproduction number estimate (R0): disease name, location, date, R0 value, %CI values, and method?'''
question_json = " {\"question\": \"What are the values for the following properties of the basic reproduction number estimate (R0): disease name, location, date, R0 value, %CI values, and method?\"}"
train_df = pd.read_excel("../../../data/raw/cord19_train_dev_test/train.xlsx").astype(object)
main_template_name = "best_templates_dataset_random_final"
result_train_json_based_path = "../../../data/processed/train_templated_files/" + main_template_name + "/flatten/json"
number_of_needed_templates = math.ceil(len(best_patterns_list) / 2)
mode = "mixed"


def save_files(path, file_name, data):
    [os.makedirs(os.path.join(*path.split("/")[:i]), exist_ok=True) for i in range(1, len(path.split("/")) + 1)]
    with open(f"{path}/{file_name}", "w") as outfile:
        outfile.write(json.dumps(data, indent=4))


def get_random_objects(seed):
    random.seed(seed)
    if number_of_needed_templates > len(best_patterns_list):
        raise ValueError("Number of random objects requested exceeds the size of the list.")
    random_indexes = random.sample(range(len(best_patterns_list)), int(number_of_needed_templates))
    random_templates = [best_patterns_list[i] for i in random_indexes]
    return random_templates


def change_data_format(data, file_type):
    global_id_vals = [str(d['instanceGlobalId']) for d in data]
    instance_id_vals = [str(d['instanceId']) for d in data]
    template_name_vals = [d['templateName'] for d in data]
    template_number_vals = [str(d['templateNumber']) for d in data]
    coord_id_vals = [d['cordId'] for d in data]
    main_coord_id_vals = [d['mainCordId'] for d in data]
    text_vals = [d['prompt'] for d in data]
    if file_type == 'text':
        label_vals = [d['response'] for d in data]
    else:
        label_vals = [json.dumps(d['response']) for d in data]
    # Create a new dictionary with the column values
    new_data = {
        'instance_global_id': global_id_vals, 'instance_id': instance_id_vals,
        'template_name': template_name_vals, 'template_number': template_number_vals,
        'cord_id': coord_id_vals, 'main_cord_id': main_coord_id_vals,
        'prompt': text_vals, 'response': label_vals  # change this based on type
    }
    return new_data


def change_dataset_formats():
    for file_path in [f"{result_train_json_based_path}/train_flatten_json_based"]:
        with open(file_path + '.json', 'r') as f:
            data = json.load(f)
        if 'text_based' in file_path:
            file_type = 'text'
        else:
            file_type = 'json'
        new_data = change_data_format(data, file_type)
        with open(file_path + '_formatted.json', 'w') as f:
            json.dump(new_data, f)


def get_filled_prompt_and_response(abstract, title, template_prompt, template_answer, json_answer):
    if "{title}" in template_prompt:
        filled_prompt = str(template_prompt).replace('{title}', title).replace('{context}', abstract).replace(
            '{question}', question)
    else:
        filled_prompt = str(template_prompt).replace('{context}', title + "\n" + abstract).replace(
            '{question}', question)

    filled_response_json = str(template_answer).replace('{answer}', json_answer).replace('{question}', question_json)

    if filled_response_json != 'unanswerable':
        filled_response_json = json.loads(filled_response_json)

    return filled_prompt, filled_response_json


def is_valid_template(template_name, template_number, mode):
    return not ((template_name == 'squad_v2' and template_number == 3) or
                (template_name == 'drop' and template_number == 8) or
                (mode == 'test' and template_name == 'drop' and (template_number == 9 or template_number == 10)))


def get_train_datasets(train_sub_folder_name):
    with open(
            '../../../data/processed/train_templated_files/' + train_sub_folder_name + '/flatten/json/train_flatten_json_based_formatted.json',
            "r") as f:
        train_data_json_based = json.load(f)

    train_dataset_json_based = Dataset.from_dict(train_data_json_based)
    return train_dataset_json_based


def build_templated_jsons(dataframe):
    instance_global_id = 1
    flat_dataset_json = []

    for index, row in dataframe.iterrows():
        main_cord_uid, cord_uid, title, abstract, json_answer = \
            row['main_cord_uid'], row['cord_uid'], row['title'], row['abstract'], row['json_response']
        if mode == "all":
            selected_templates = best_patterns_list
        else:
            selected_templates = get_random_objects(index)
        for template in selected_templates:

            template_name, template_number, template_prompt, template_answer = template[0], template[1], template[2], \
                template[3]
            if is_valid_template(template_name, template_number, "train"):
                filled_prompt, filled_response_json = get_filled_prompt_and_response(abstract,
                                                                                     title,
                                                                                     template_prompt,
                                                                                     template_answer,
                                                                                     json_answer)

                flat_data_dict_json = {'instanceGlobalId': instance_global_id, 'instanceId': index + 1,
                                       'templateName': template_name, 'templateNumber': template_number,
                                       'cordId': cord_uid, 'mainCordId': main_cord_uid, 'prompt': filled_prompt,
                                       'response': filled_response_json}

                flat_dataset_json.append(flat_data_dict_json)
                instance_global_id += 1

    return flat_dataset_json


def save_train_files():
    flat_dataset_json = build_templated_jsons(dataframe=train_df)
    save_files(path=result_train_json_based_path, file_name="train_flatten_json_based.json", data=flat_dataset_json)


def get_hg_dataset(dev_dataset, test_dataset, train_data):
    return DatasetDict({
        'train': train_data,
        'dev': dev_dataset,
        'test': test_dataset
    })


def save_hg_dataset(path, dataset):
    [os.makedirs(os.path.join(*path.split("/")[:i]), exist_ok=True) for i in range(1, len(path.split("/")) + 1)]
    dataset.save_to_disk(path)


def get_dev_test_datasets():

    with open('../../../data/processed/dev_templated_files/flatten/json/dev_flatten_json_based_formatted.json',
              "r") as f:
        dev_data_json_based = json.load(f)

    with open('../../../data/processed/test_templated_files/flatten/json/test_flatten_json_based_formatted.json',
              "r") as f:
        test_data_json_based = json.load(f)
    dev_dataset_json_based = Dataset.from_dict(dev_data_json_based)
    test_dataset_json_based = Dataset.from_dict(test_data_json_based)
    return dev_dataset_json_based, test_dataset_json_based


def build_final_datasets():
    dev_dataset_json_based, test_dataset_json_based = get_dev_test_datasets()
    train_dataset_json_based = get_train_datasets(main_template_name)
    full_dataset_json_based = get_hg_dataset(dev_dataset_json_based, test_dataset_json_based, train_dataset_json_based)
    save_hg_dataset(
        '../../../data/processed/final_datasets/' + main_template_name + '/json_based/orkg_brp_dataset',
        full_dataset_json_based)
    print(full_dataset_json_based)


def get_desired_dataset():
    save_train_files()
    change_dataset_formats()
    build_final_datasets()


get_desired_dataset()
