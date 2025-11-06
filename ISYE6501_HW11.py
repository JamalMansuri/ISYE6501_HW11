# 1. Imports and data load
import pulp as lp
import pandas as pd
import os
import glob

script_dir = os.path.dirname(os.path.abspath(__file__))
excel_files = glob.glob(os.path.join(script_dir, "*.xlsx"))
print(f"Found {len(excel_files)} files:", excel_files)

# assume: first is large, second is small
diet_large = pd.read_excel(excel_files[0], sheet_name='Sheet1')
diet_small = pd.read_excel(excel_files[1], sheet_name='Sheet1')

# 2. Prep small diet table (last 2 rows = min/max)
min_req = diet_small.iloc[-2]
max_req = diet_small.iloc[-1]
foods_df = diet_small.iloc[:-2].copy()

price_col = 'Price/ Serving'
foods_df = foods_df[foods_df[price_col].notna()].copy()
foods_df = foods_df.set_index('Foods')
food_names = list(foods_df.index)

nutrients = [
    'Calories', 'Cholesterol mg', 'Total_Fat g', 'Sodium mg',
    'Carbohydrates g', 'Dietary_Fiber g', 'Protein g'
]

# 3. Basic model: cheapest diet
diet_prob = lp.LpProblem("Diet_Problem", lp.LpMinimize)

x = lp.LpVariable.dicts("servings", food_names, lowBound=0)

diet_prob += lp.lpSum([x[f] * float(foods_df.at[f, price_col]) for f in food_names])

for nutr in nutrients:
    diet_prob += lp.lpSum([x[f] * float(foods_df.at[f, nutr]) for f in food_names]) >= float(min_req[nutr])
    diet_prob += lp.lpSum([x[f] * float(foods_df.at[f, nutr]) for f in food_names]) <= float(max_req[nutr])

diet_prob.solve(lp.PULP_CBC_CMD(msg=False))

print("--- Basic diet ---")
print("Status:", lp.LpStatus[diet_prob.status])
for f in food_names:
    val = x[f].value()
    if val and val > 1e-5:
        print(f"{f}: {val:.3f}")

# 4. Extended model: with on/off choice
diet_prob2 = lp.LpProblem("Diet_Problem_with_choices", lp.LpMinimize)

x2 = lp.LpVariable.dicts("servings", food_names, lowBound=0)
y2 = lp.LpVariable.dicts("choose", food_names, lowBound=0, upBound=1, cat='Binary')

diet_prob2 += lp.lpSum([x2[f] * float(foods_df.at[f, price_col]) for f in food_names])

BIG_M = 10
for f in food_names:
    diet_prob2 += x2[f] >= 0.1 * y2[f]
    diet_prob2 += x2[f] <= BIG_M * y2[f]

for nutr in nutrients:
    diet_prob2 += lp.lpSum([x2[f] * float(foods_df.at[f, nutr]) for f in food_names]) >= float(min_req[nutr])
    diet_prob2 += lp.lpSum([x2[f] * float(foods_df.at[f, nutr]) for f in food_names]) <= float(max_req[nutr])

if 'Celery, Raw' in food_names and 'Frozen Broccoli' in food_names:
    diet_prob2 += y2['Celery, Raw'] + y2['Frozen Broccoli'] <= 1

diet_prob2.solve(lp.PULP_CBC_CMD(msg=False))

print("\n--- Extended diet ---")
print("Status:", lp.LpStatus[diet_prob2.status])
for f in food_names:
    val = x2[f].value()
    if val and val > 1e-5:
        print(f"{f}: {val:.3f} (choose={int(y2[f].value())})")
print("Total cost:", round(lp.value(diet_prob2.objective), 2))