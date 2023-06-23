import pandas as pd

data = pd.read_excel("../../../data/raw/initial_dataset/initial_filtered_cord_data/filtered_metadata.xlsx")

data.drop_duplicates(subset=['cord_uid', 'abstract'], keep="first", inplace=True)
data.drop_duplicates(subset='abstract', keep="first", inplace=True)

data.to_excel("../../../data/raw/initial_dataset/cleaned_cord_data/cleaned_cord_data_exact_duplicates.xlsx",
              index=False)
