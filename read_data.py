import pandas as pd
from pprint import pprint

df = pd.read_parquet("/home/ubuntu/Working/neurips/code_train.parquet?download=true")

for i in range(2):
    print("="*80, "row", i, "="*80)
    pprint(df.iloc[i].to_dict(), width=120)