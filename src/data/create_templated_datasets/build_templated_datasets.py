import json
import os

import pandas as pd
from datasets import Dataset, DatasetDict

# note: don't use template 3 from squad_v2 and template 8 from drop generally
# don't use drop 9,10 in test and dev
# reference to """Templates for FLAN.""" at :https://github.com/google-research/FLAN/blob/main/flan/templates.py
PATTERNS = {
    "squad_v2": [
        ("{title}:\n\n{context}\n\nPlease answer a question about this article. If the question is unanswerable, say \"unanswerable\". {question}", "{answer}"),
        ("Read this and answer the question. If the question is unanswerable, say \"unanswerable\".\n\n{context}\n\n{question}", "{answer}"),
        ("What is a question about this article? If the question is unanswerable, say \"unanswerable\".\n\n{context}\n\n{question}", "{answer}"),
        ("{context}\n{question} (If the question is unanswerable, say \"unanswerable\")", "{answer}"),
        ("{context}\nTry to answer this question if possible (otherwise reply \"unanswerable\"): {question}", "{answer}"),
        ("{context}\nIf it is possible to answer this question, answer it for me (else, reply \"unanswerable\"): {question}", "{answer}"),
        ("{context}\n\nAnswer this question, if possible (if impossible, reply \"unanswerable\"): {question}", "{answer}"),
        ("Read this: {context}\n\n{question}\nWhat is the answer? (If it cannot be answered, return \"unanswerable\")", "{answer}"),
        ("Read this: {context}\nNow answer this question, if there is an answer (If it cannot be answered, return \"unanswerable\"): {question}", "{answer}"),
        ("{context}\nIs there an answer to this question (If it cannot be answered, say \"unanswerable\"): {question}", "{answer}"),
    ],
    "drop": [
        ("Answer based on context:\n\n{context}\n\n{question}", "{answer}"),
        ("{context}\n\nAnswer this question based on the article: {question}", "{answer}"),
        ("{context}\n\n{question}", "{answer}"),
        ("{context}\nAnswer this question: {question}", "{answer}"),
        ("Read this article and answer this question {context}\n{question}", "{answer}"),
        ("{context}\n\nBased on the above article, answer a question. {question}", "{answer}"),
        ("Context: {context}\n\nQuestion: {question}\n\nAnswer:", "{answer}"),
        ("Write an article that answers the following question: {question}", "{context}"),
        ("Write a question about the following article: {context}", "{question}"),
        ("{context}\n\nAsk a question about this article.", "{question}"),
    ]

}

question = '''What are the values for the following properties of the basic reproduction number estimate (R0): disease name, location, date, R0 value, %CI values, and method?'''
question_json = " {\"question\": \"What are the values for the following properties of the basic reproduction number estimate (R0): disease name, location, date, R0 value, %CI values, and method?\"}"

train_df = pd.read_excel("../../../data/raw/cord19_train_dev_test/train.xlsx").astype(object)
test_df = pd.read_excel("../../../data/raw/cord19_train_dev_test/test.xlsx").astype(object)
dev_df = pd.read_excel("../../../data/raw/cord19_train_dev_test/dev.xlsx").astype(object)

full_path_list_of_flats = []


def is_valid_template(template_name, template_number, mode):
    return not ((template_name == 'squad_v2' and template_number == 3) or
                (template_name == 'drop' and template_number == 8) or
                (mode == 'test' and template_name == 'drop' and (template_number == 9 or template_number == 10)))


def get_filled_json_and_text_response(text_answer, json_answer, temp_second_part):
    filled_response_text = str(temp_second_part).replace('{answer}', text_answer).replace(
        '{question}', question)
    filled_response_json = str(temp_second_part).replace('{answer}', json_answer).replace(
        '{question}', question_json)

    if filled_response_json != 'unanswerable':
        filled_response_json = json.loads(filled_response_json)

    return filled_response_json, filled_response_text


def get_filled_prompt(abstract, title, temp_first_part):
    if "{title}" in temp_first_part:
        return str(temp_first_part).replace('{title}', title).replace('{context}', abstract).replace(
            '{question}', question)
    else:
        return str(temp_first_part).replace('{context}', title + "\n" + abstract).replace(
            '{question}', question)


def build_templated_jsons(dataframe, desired_template_name_list, desired_template_number_list, mode):
    instance_global_id = 1
    template_filled_nested_text_main = []
    template_filled_nested_json_main = []
    flat_dataset_text = []
    flat_dataset_json = []

    for template_name in desired_template_name_list:
        for template_number in desired_template_number_list:

            if is_valid_template(template_name, template_number, mode):

                template_tuple = PATTERNS[template_name][template_number - 1]
                nested_data_dict_text_list = []
                nested_data_dict_json_list = []
                for index, row in dataframe.iterrows():
                    main_cord_uid, cord_uid, title, abstract, text_answer, json_answer = \
                        row['main_cord_uid'], row['cord_uid'], row['title'], row['abstract'], row['text_response'], row[
                            'json_response']
                    filled_prompt, filled_response_json, filled_response_text = get_filled_prompt_and_response(abstract,
                                                                                                               title,
                                                                                                               template_tuple,
                                                                                                               text_answer,
                                                                                                               json_answer)

                    nested_data_dict_text = {'instanceGlobalId': instance_global_id, 'instanceId': index + 1,
                                             'cordId': cord_uid, 'mainCordId': main_cord_uid, 'prompt': filled_prompt,
                                             'response': filled_response_text}
                    nested_data_dict_json = {'instanceGlobalId': instance_global_id, 'instanceId': index + 1,
                                             'cordId': cord_uid, 'mainCordId': main_cord_uid, 'prompt': filled_prompt,
                                             'response': filled_response_json}

                    flat_data_dict_text = {'instanceGlobalId': instance_global_id, 'instanceId': index + 1,
                                           'templateName': template_name, 'templateNumber': template_number,
                                           'cordId': cord_uid, 'mainCordId': main_cord_uid, 'prompt': filled_prompt,
                                           'response': filled_response_text}

                    flat_data_dict_json = {'instanceGlobalId': instance_global_id, 'instanceId': index + 1,
                                           'templateName': template_name, 'templateNumber': template_number,
                                           'cordId': cord_uid, 'mainCordId': main_cord_uid, 'prompt': filled_prompt,
                                           'response': filled_response_json}

                    nested_data_dict_text_list.append(nested_data_dict_text)
                    flat_dataset_text.append(flat_data_dict_text)

                    nested_data_dict_json_list.append(nested_data_dict_json)
                    flat_dataset_json.append(flat_data_dict_json)
                    instance_global_id += 1

                template_filled_dict_text = {'templateName': template_name, 'templateNumber': template_number,
                                             'dataList': nested_data_dict_text_list}
                template_filled_nested_text_main.append(template_filled_dict_text)

                template_filled_dict_json = {'templateName': template_name, 'templateNumber': template_number,
                                             'dataList': nested_data_dict_json_list}
                template_filled_nested_json_main.append(template_filled_dict_json)

    nested_dataset_text = {'dataset': template_filled_nested_text_main}
    nested_dataset_json = {'dataset': template_filled_nested_json_main}

    return flat_dataset_text, flat_dataset_json, nested_dataset_text, nested_dataset_json


def get_filled_prompt_and_response(abstract, title, template_tuple, text_answer, json_answer):
    filled_prompt = get_filled_prompt(abstract, title, template_tuple[0])
    filled_response_json, filled_response_text = get_filled_json_and_text_response(
        text_answer, json_answer, template_tuple[1])
    return filled_prompt, filled_response_json, filled_response_text


def get_dev_or_test_templated_datasets(dataframe):
    template_names = ["squad_v2", "drop"]
    template_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    return build_templated_jsons(dataframe, template_names, template_numbers, 'test')


def build_train_templates_json_files(template_names, template_numbers, template_path_name):
    flat_dataset_text_train, flat_dataset_json_train, nested_dataset_text_train, nested_dataset_json_train = build_templated_jsons(
        train_df, template_names, template_numbers, 'train')
    if all(lst for lst in
           [flat_dataset_text_train, flat_dataset_json_train, nested_dataset_text_train, nested_dataset_json_train] if
           lst):
        save_files("../../../data/processed/train_templated_files/" + template_path_name + "/flatten/text",
                   "train_flatten_text_based",
                   flat_dataset_text_train)
        save_files("../../../data/processed/train_templated_files/" + template_path_name + "/flatten/json",
                   "train_flatten_json_based",
                   flat_dataset_json_train)
        save_files("../../../data/processed/train_templated_files/" + template_path_name + "/nested/text",
                   "train_nested_text_based",
                   nested_dataset_text_train)
        save_files("../../../data/processed/train_templated_files/" + template_path_name + "/nested/json",
                   "train_nested_json_based",
                   nested_dataset_json_train)


def save_train_files():
    template_names = ["squad_v2", "drop"]
    template_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    build_train_templates_json_files(template_names, template_numbers, "all_18_templates")

    for name in template_names:
        for number in template_numbers:
            if not ((name == 'squad_v2' and number == 3) or (
                    name == 'drop' and (number == 8 or number == 9 or number == 10))):
                build_train_templates_json_files([name], [number], name + "_" + str(number))


def get_hg_dataset(dev_dataset_text_based, test_dataset_text_based, train_data_text_based):
    return DatasetDict({
        'train': train_data_text_based,
        'dev': dev_dataset_text_based,
        'test': test_dataset_text_based
    })


def get_train_datasets(train_sub_folder_name):
    with open(
            '../../../data/processed/train_templated_files/' + train_sub_folder_name + '/flatten/text/train_flatten_text_based_formatted.json',
            "r") as f:
        train_data_text_based = json.load(f)
    with open(
            '../../../data/processed/train_templated_files/' + train_sub_folder_name + '/flatten/json/train_flatten_json_based_formatted.json',
            "r") as f:
        train_data_json_based = json.load(f)

    train_dataset_text_based = Dataset.from_dict(train_data_text_based)
    train_dataset_json_based = Dataset.from_dict(train_data_json_based)

    return train_dataset_json_based, train_dataset_text_based


def get_dev_test_datasets():
    with open('../../../data/processed/dev_templated_files/flatten/text/dev_flatten_text_based_formatted.json',
              "r") as f:
        dev_data_text_based = json.load(f)
    with open('../../../data/processed/dev_templated_files/flatten/json/dev_flatten_json_based_formatted.json',
              "r") as f:
        dev_data_json_based = json.load(f)
    with open('../../../data/processed/test_templated_files/flatten/text/test_flatten_text_based_formatted.json',
              "r") as f:
        test_data_text_based = json.load(f)
    with open('../../../data/processed/test_templated_files/flatten/json/test_flatten_json_based_formatted.json',
              "r") as f:
        test_data_json_based = json.load(f)
    dev_dataset_text_based = Dataset.from_dict(dev_data_text_based)
    dev_dataset_json_based = Dataset.from_dict(dev_data_json_based)
    test_dataset_text_based = Dataset.from_dict(test_data_text_based)
    test_dataset_json_based = Dataset.from_dict(test_data_json_based)
    return dev_dataset_json_based, dev_dataset_text_based, test_dataset_json_based, test_dataset_text_based


def change_dataset_formats(path_list_param):
    for file_path in path_list_param:
        with open(file_path + '.json', 'r') as f:
            data = json.load(f)
        if 'text_based' in file_path:
            file_type = 'text'
        else:
            file_type = 'json'
        new_data = change_data_format(data, file_type)
        with open(file_path + '_formatted.json', 'w') as f:
            json.dump(new_data, f)


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


def save_test_files():
    flat_dataset_text_test, flat_dataset_json_test, nested_dataset_text_test, nested_dataset_json_test = get_dev_or_test_templated_datasets(
        test_df)
    save_files("../../../data/processed/test_templated_files/flatten/text", "test_flatten_text_based",
               flat_dataset_text_test)
    save_files("../../../data/processed/test_templated_files/flatten/json", "test_flatten_json_based",
               flat_dataset_json_test)
    save_files("../../../data/processed/test_templated_files/nested/text", "test_nested_text_based",
               nested_dataset_text_test)
    save_files("../../../data/processed/test_templated_files/nested/json", "test_nested_json_based",
               nested_dataset_json_test)


def save_dev_files():
    flat_dataset_text_dev, flat_dataset_json_dev, nested_dataset_text_dev, nested_dataset_json_dev = get_dev_or_test_templated_datasets(
        dev_df)

    save_files("../../../data/processed/dev_templated_files/flatten/text", "dev_flatten_text_based",
               flat_dataset_text_dev)
    save_files("../../../data/processed/dev_templated_files/flatten/json", "dev_flatten_json_based",
               flat_dataset_json_dev)
    save_files("../../../data/processed/dev_templated_files/nested/text", "dev_nested_text_based",
               nested_dataset_text_dev)
    save_files("../../../data/processed/dev_templated_files/nested/json", "dev_nested_json_based",
               nested_dataset_json_dev)


def save_files(path, file_name, data):
    [os.makedirs(os.path.join(*path.split("/")[:i]), exist_ok=True) for i in range(1, len(path.split("/")) + 1)]
    with open(path + "/" + file_name + ".json", "w") as outfile:
        outfile.write(json.dumps(data, indent=4))
    if 'flatten' in path:
        full_path_list_of_flats.append(path + "/" + file_name)


def save_hg_dataset(path, dataset):
    [os.makedirs(os.path.join(*path.split("/")[:i]), exist_ok=True) for i in range(1, len(path.split("/")) + 1)]
    dataset.save_to_disk(path)


def build_final_datasets():
    dev_dataset_json_based, dev_dataset_text_based, test_dataset_json_based, test_dataset_text_based = get_dev_test_datasets()
    subdirectories_list = [dir_param for root, dirs, files in os.walk("../../../data/processed/train_templated_files")
                           for dir_param in dirs if
                           root == "../../../data/processed/train_templated_files"]
    for train_sub_folder_name in subdirectories_list:
        train_dataset_json_based, train_dataset_text_based = get_train_datasets(train_sub_folder_name)

        full_dataset_text_based = get_hg_dataset(dev_dataset_text_based, test_dataset_text_based,
                                                 train_dataset_text_based)

        full_dataset_json_based = get_hg_dataset(dev_dataset_json_based, test_dataset_json_based,
                                                 train_dataset_json_based)

        save_hg_dataset(
            '../../../data/processed/final_datasets/' + train_sub_folder_name + '/text_based/orkg_brp_dataset',
            full_dataset_text_based)
        save_hg_dataset(
            '../../../data/processed/final_datasets/' + train_sub_folder_name + '/json_based/orkg_brp_dataset',
            full_dataset_json_based)


def get_desired_dataset():
    save_dev_files()
    save_test_files()
    save_train_files()
    change_dataset_formats(full_path_list_of_flats)
    build_final_datasets()


get_desired_dataset()
