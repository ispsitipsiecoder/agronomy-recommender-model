import pandas as pd

df = pd.read_csv('../data/raw/crop_recommendation.csv')
paddy = df[df['label'] == 'rice'].copy().reset_index(drop=True)

print(f"Total rows: {len(df)}")
print(f"Paddy rows: {len(paddy)}")
print(f"Columns: {paddy.columns.tolist()}")
print(paddy.describe().round(2))
