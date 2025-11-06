#########
# Git commands to update this file 
#########

#cd "/Users/jamal/Library/CloudStorage/GoogleDrive-jamal.mansuri0@gmail.com/My Drive/Gtech_OMSA/ISYE_6501/Module 15 & 16"
# git add ISYE6501_HW11.py
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


## Load Data
script_dir = os.path.dirname(os.path.abspath(__file__))

diet_data = glob.glob(os.path.join(script_dir, "*.xlsx"))

print(f"Found {len(diet_data)} files:", diet_data)

diet_large = pd.read_excel(diet_data[0], sheet_name = 'Sheet1')
diet = pd.read_excel(diet_data[1], sheet_name = 'Sheet1')

## Initial Exploration
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


## Fixing Headers
diet_large.columns = diet_large.iloc[0]
diet_large = diet_large[1:]


# --- Simple diet model on the small dataset ---
# Last two rows are min and max requirements
min_req = diet.iloc[-2]
max_req = diet.iloc[-1]
foods_df = diet.iloc[:-2].copy()
# make Foods the index so lookups never return empty

cost_col = 'Price/ Serving'
# drop rows that don't have a price
foods_df = foods_df[foods_df[cost_col].notna()].copy()

foods_df = foods_df.set_index('Foods')

# Create LP to minimize cost
diet_prob = lp.LpProblem("Diet_Problem", lp.LpMinimize)
food_names = list(foods_df.index)

# decision variables: servings of each food
x = lp.LpVariable.dicts("servings", food_names, lowBound=0)

# objective: minimize total cost

diet_prob += lp.lpSum([
    x[f] * float(foods_df.at[f, cost_col])
    for f in food_names
])

nutrient_cols = ['Calories', 'Cholesterol mg', 'Total_Fat g', 'Sodium mg',
                 'Carbohydrates g', 'Dietary_Fiber g', 'Protein g']

for nutr in nutrient_cols:
    diet_prob += lp.lpSum([
        x[f] * float(foods_df.at[f, nutr])
        for f in food_names
    ]) >= float(min_req[nutr])
    diet_prob += lp.lpSum([
        x[f] * float(foods_df.at[f, nutr])
        for f in food_names
    ]) <= float(max_req[nutr])

# solve
diet_prob.solve(lp.PULP_CBC_CMD(msg=False))

print("Status:", lp.LpStatus[diet_prob.status])
print("Chosen foods:")
for f in food_names:
    val = x[f].value()
    if val is not None and val > 1e-5:
        print(f"{f}: {val:.3f} servings")


 # --- Extended model with binary choices ---
 # create a new problem
diet_prob2 = lp.LpProblem("Diet_Problem_with_choices", lp.LpMinimize)

# decision vars: servings (continuous) and choose (binary)
x2 = lp.LpVariable.dicts("servings", food_names, lowBound=0)
y2 = lp.LpVariable.dicts("choose",   food_names, lowBound=0, upBound=1, cat='Binary')

# objective: same cost
diet_prob2 += lp.lpSum([
    x2[f] * float(foods_df.at[f, cost_col])
    for f in food_names
])

BIG_M = 10
for f in food_names:
    # link: if chosen then at least 0.1, else 0
    diet_prob2 += x2[f] >= 0.1 * y2[f]
    diet_prob2 += x2[f] <= BIG_M * y2[f]

# nutrition constraints (same as before)
for nutr in nutrient_cols:
    diet_prob2 += lp.lpSum([
        x2[f] * float(foods_df.at[f, nutr])
        for f in food_names
    ]) >= float(min_req[nutr])
    diet_prob2 += lp.lpSum([
        x2[f] * float(foods_df.at[f, nutr])
        for f in food_names
    ]) <= float(max_req[nutr])

# optional: mutual exclusion example (adjust names to match your file)
if 'Celery, Raw' in food_names and 'Frozen Broccoli' in food_names:
    diet_prob2 += y2['Celery, Raw'] + y2['Frozen Broccoli'] <= 1

# solve second model
diet_prob2.solve(lp.PULP_CBC_CMD(msg=False))

print("\n--- Extended model (with binaries) ---")
print("Status:", lp.LpStatus[diet_prob2.status])
for f in food_names:
    val = x2[f].value()
    if val is not None and val > 1e-5:
        print(f"{f}: {val:.3f} servings (chosen={int(y2[f].value())})")
print("Total cost (extended):", round(lp.value(diet_prob2.objective), 2))