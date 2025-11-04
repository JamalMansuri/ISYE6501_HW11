#########
# Git commands to update this file 
#########

#git add ISYE6501_HW11.py
#git commit -m "Update HW11"
#git push

#########
# Initialization 
#########

import pulp as lp
import pandas as pd
import numpy as np
import os
import glob

script_dir = os.path.dirname(os.path.abspath(__file__))

diet_data = glob.glob(os.path.join(script_dir, "*.xlsx"))

print(f"Found {len(diet_data)} files:", diet_data)

diet_large = pd.read_excel(diet_data[0], sheet_name = 'Sheet1')
diet = pd.read_excel(diet_data[1], sheet_name = 'Sheet1')

print(diet_large.head())
print(diet.head())

diet_large_columns = diet_large.columns.tolist()
diet_columns = diet.columns.tolist()
print("Columns in large dataset:", diet_large_columns)
print("Columns in small dataset:", diet_columns)

diet_large_desc = diet_large.describe()
diet_desc = diet.describe()
print("Description of large dataset:\n", diet_large_desc)
print("Description of small dataset:\n", diet_desc)

diet_large.columns = diet_large.iloc[0]
diet_large = diet_large[1:]

print("Columns in large dataset after setting header:", diet_large.columns.tolist())
print(diet_large.head())