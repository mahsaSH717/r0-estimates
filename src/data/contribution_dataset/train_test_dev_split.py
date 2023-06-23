import math

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

input_data_file_path = '../../../data/raw/initial_dataset/contribution_based_dataset/group_by_dataset.xlsx'
train_set_path = '../../../data/raw/cord19_train_dev_test/train.xlsx'
test_set_path = '../../../data/raw/cord19_train_dev_test/test.xlsx'
dev_set_path = '../../../data/raw/cord19_train_dev_test/dev.xlsx'

random_state = 117
test_proportion = 0.2
dev_proportion = 0.1

main_df = pd.read_excel(input_data_file_path)
main_df = main_df.astype(object)


def split_data(main_data_frame, test_proportion_param, train_proportion_param, random_state_param):
    total_count = len(main_data_frame)
    unique_main_cord_ids = main_data_frame['main_cord_uid'].unique()
    print("TOTAL:", total_count)
    print("TOTAL UNIQUE:", len(unique_main_cord_ids))
    yes_portion = main_data_frame.query('annotator_investigating_R0 != -1')
    no_portion = main_data_frame.query('annotator_investigating_R0 == -1')
    unique_main_yes_cord_ids = yes_portion['main_cord_uid'].unique()
    print("YES:", len(yes_portion))
    print("YES UNIQUE:", len(unique_main_yes_cord_ids))
    unique_main_no_cord_ids = no_portion['main_cord_uid'].unique()
    print("NO:", len(no_portion))
    print("NO UNIQUE:", len(unique_main_no_cord_ids))

    train_size = math.ceil(train_proportion_param * total_count)
    test_size = total_count - train_size
    print("Total Train:", train_size)
    print("Total Test:", test_size)
    splitter = GroupShuffleSplit(test_size=test_proportion_param, n_splits=1, random_state=random_state_param)
    split = splitter.split(yes_portion, groups=yes_portion['main_cord_uid'])
    train_yes_inds, test_yes_inds = next(split)
    train_yes = yes_portion.iloc[train_yes_inds]
    test_yes = yes_portion.iloc[test_yes_inds]
    print("Yes Part Split:")
    print("Yes Part Train:", len(train_yes_inds))
    print("Yes Part Test:", len(test_yes_inds))
    check1 = np.in1d(train_yes_inds, test_yes_inds)
    print(np.all(check1 == 0))
    no_portion_train_size = train_size - len(train_yes_inds)
    no_portion_test_size = test_size - len(test_yes_inds)
    splitter2 = GroupShuffleSplit(test_size=no_portion_test_size, n_splits=1, random_state=random_state_param)
    split2 = splitter2.split(no_portion, groups=no_portion['main_cord_uid'])
    train_no_inds, test_no_inds = next(split2)
    train_no = no_portion.iloc[train_no_inds]
    test_no = no_portion.iloc[test_no_inds]
    print("No Part Split:")
    print("No Part Train:", len(train_no_inds))
    print("No Part Test:", len(test_no_inds))
    check2 = np.in1d(train_no_inds, test_no_inds)
    print(np.all(check2 == 0))
    train = pd.concat([train_yes, train_no])
    test = pd.concat([test_yes, test_no])

    train_set = train.sample(frac=1, random_state=random_state_param)
    test_set = test.sample(frac=1, random_state=random_state_param)
    print("_____________Split is Finished________________")
    return train_set, test_set


train_dev_df, test_df = split_data(main_data_frame=main_df, test_proportion_param=test_proportion,
                                   train_proportion_param=1 - test_proportion, random_state_param=random_state)

train_df, dev_df = split_data(main_data_frame=train_dev_df, test_proportion_param=dev_proportion,
                              train_proportion_param=1 - dev_proportion, random_state_param=random_state)

# Save train, dev, and test sets to Excel files
train_df.to_excel(train_set_path, index=False)
test_df.to_excel(test_set_path, index=False)
dev_df.to_excel(dev_set_path, index=False)
