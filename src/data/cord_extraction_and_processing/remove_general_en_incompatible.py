import pandas as pd

df = pd.read_excel(
    "../../../data/raw/initial_dataset/cleaned_cord_data/cord_data_removed_fuzzy_duplicates_manually.xlsx")
df = df.astype(object)

for ind in df.index:
    if (df['abstract'][ind]).count(".") <= 0:
        df.drop(ind, axis=0, inplace=True)

df.to_excel("../../../data/raw/initial_dataset/cleaned_cord_data/cord_data_cleaned_final.xlsx", index=False)
