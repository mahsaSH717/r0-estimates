import pandas as pd
from thefuzz import fuzz

cluster_dic = {}
filtered_cluster = {}


def is_same_research_data(rd1, rd2):
    return fuzz.ratio(rd1['abstract'], rd2['abstract']) > 95


def exists_in_cluster(index):
    for key_param, value_param in cluster_dic.items():
        if key_param is index:
            return True
        elif value_param.count(index) > 0:
            return True
        else:
            return False


def find_cluster(index):
    found_key = ""
    for key in cluster_dic:
        if is_same_research_data(df.loc[int(key)], df.loc[int(index)]):
            found_key = key
            break
    if found_key != "":
        return found_key
    else:
        return str(index)


def find_cluster_for_value(index):
    if index in filtered_cluster.keys():
        return index
    else:
        for value in filtered_cluster.values():
            if value.count(index) > 0:
                return value[0]
    return None


def my_filtering_function(pair):
    key, value = pair
    if len(value) <= 1:
        return False  # filter pair out of the dictionary
    else:
        return True  # keep pair in the filtered dictionary


df = pd.read_excel("../../../data/raw/initial_dataset/cleaned_cord_data/cleaned_cord_data_exact_duplicates.xlsx")
df = df.astype(object)

for i in range(len(df)):
    if not exists_in_cluster(str(i)):

        key_to_append_or_insert = find_cluster(str(i))
        if key_to_append_or_insert in cluster_dic.keys():
            cluster_dic[key_to_append_or_insert].append(str(i))
        else:
            cluster_dic[key_to_append_or_insert] = [key_to_append_or_insert]

filtered_cluster = dict(filter(my_filtering_function, cluster_dic.items()))

for m in range(len(df)):
    df.loc[m, 'cluster_id'] = find_cluster_for_value(str(m))

df.to_excel("../../../data/raw/initial_dataset/cleaned_cord_data/cord_data_fuzzy_clustered.xlsx")
