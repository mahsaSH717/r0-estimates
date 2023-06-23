import csv
import re
from collections import defaultdict

import pandas as pd

cord_uid_to_text = defaultdict(list)
data = []
errData = []

regex1 = r'(reproductive number estimate)|(reproductive number)| ' \
         r'(R0 number) |(R0 estimate)|(r0)|(r\(0\))|(reproductive estimate)|' \
         r'(reproduction number)|(reproduction estimate)|' \
         r'(reproduction ratio)|(reproductive rate)'
regex2 = r'[1-9]'

with open('../../../data/raw/initial_dataset/cord_2022-06-02_metadata/metadata.csv') as f_in:
    reader = csv.DictReader(f_in)
    while True:
        try:
            row = next(reader)
            title = str(row['title'])
            abstract = row['abstract']
            if re.search(regex1, abstract, re.I):
                if re.search(regex2, abstract):
                    data.append([row['cord_uid'], row['title'], row['abstract'], row['publish_time']])
        except StopIteration:
            break
        except Exception as e:
            errData.append([reader.line_num, type(e).__name__])

    print("End of file!\n")
    print("Data Count", len(data))

    df = pd.DataFrame(data, columns=['cord_uid', 'title', 'abstract', 'publishTime'])
    df_error = pd.DataFrame(errData, columns=['err_line', 'err_type'])
    df.to_excel('../../../data/raw/initial_dataset/initial_filtered_cord_data/filtered_metadata.xlsx', index=False)
    df_error.to_excel('../../../data/raw/initial_dataset/initial_filtered_cord_data/failed_records_in_filtering.xlsx',
                      index=False)
