# ISYE6501_HW11.py
import pulp as lp
import pandas as pd
import os
import glob

script_dir = os.path.dirname(os.path.abspath(__file__))
excel_files = glob.glob(os.path.join(script_dir, "*.xlsx"))
print(f"Found {len(excel_files)} files:", excel_files)

diet_large = pd.read_excel(excel_files[0], sheet_name='Sheet1')
diet_small = pd.read_excel(excel_files[1], sheet_name='Sheet1')


diet_large_head = diet_large.head(3)
diet_small_head = diet_small.head(3)
print("Large diet table head:")
print(diet_large_head)
print("\nSmall diet table head:")
print(diet_small_head)

diet_large_tail = diet_large.tail(3)
diet_small_tail = diet_small.tail(3)

print("Large diet table tail:")
print(diet_large_tail)
print("\nSmall diet table tail:")
print(diet_small_tail)

diet_large_desc = diet_large.describe()
diet_small_desc = diet_small.describe()
print("\nLarge diet table description:")
print(diet_large_desc)
print("\nSmall diet table description:")
print(diet_small_desc)


diet_large.columns = [str(c).strip() for c in diet_large.columns]
diet_small.columns = [str(c).strip() for c in diet_small.columns]

if 'Unnamed: 0' in diet_large.columns:
    diet_large = diet_large.rename(columns={'Unnamed: 0': 'Food_Desc'})

# promote first data row to headers if it contains labels (e.g., Long_Desc, Protein, Cholesterol)
if str(diet_large.iloc[0, 0]).strip() == 'Long_Desc':
    _hdr = diet_large.iloc[0].astype(str).str.strip().tolist()
    # make header names unique (Energy, Energy_2, etc.)
    _seen = {}
    _uniq = []
    for name in _hdr:
        base = name
        if base in _seen:
            _seen[base] += 1
            name = f"{base}_{_seen[base]}"
        else:
            _seen[base] = 1
        _uniq.append(name)
    diet_large = diet_large.iloc[1:].copy()
    diet_large.columns = _uniq
    if 'Long_Desc' in diet_large.columns:
        diet_large = diet_large.rename(columns={'Long_Desc': 'Food_Desc'})

# got some non-food rows at the end; removed manually
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

# 5. Extra homework part: run on large diet file and minimize cholesterol

_dlc = diet_large.copy()
_dlc = _dlc.dropna(how='all').reset_index(drop=True)


if 'Food_Desc' in _dlc.columns:
    _dlc = _dlc.set_index('Food_Desc')
    _dlc = _dlc[~_dlc.index.duplicated(keep='first')]


_chol_candidates = [c for c in _dlc.columns if 'chol' in str(c).lower()]
if _chol_candidates:
    _chol_col = _chol_candidates[0]
elif '.27' in _dlc.columns:
    _chol_col = '.27'
elif ' .27' in _dlc.columns:
    _chol_col = ' .27'
else:
    
    print("[Large file] Could not identify a cholesterol column. Columns:", list(_dlc.columns))
    _chol_col = None

if _chol_col is not None:
    _dlc[_chol_col] = pd.to_numeric(_dlc[_chol_col], errors='coerce')
    _dlc = _dlc[_dlc[_chol_col].notna()]
    valid_foods = []
    for idx in _dlc.index:
        val = _dlc.at[idx, _chol_col]
        if pd.notna(val):
            valid_foods.append(idx)

    large_prob = lp.LpProblem("Diet_Large_MinChol", lp.LpMinimize)
    xl = lp.LpVariable.dicts("servings_large", valid_foods, lowBound=0)
    yl = lp.LpVariable.dicts("choose_large", valid_foods, lowBound=0, upBound=1, cat='Binary')

    
    obj_terms = []
    for f in valid_foods:
        chol_val = float(_dlc.at[f, _chol_col])
        obj_terms.append(xl[f] * chol_val)
    large_prob += lp.lpSum(obj_terms)

    
    BIG_M_L = 10
    for f in valid_foods:
        large_prob += xl[f] >= 0.1 * yl[f]
        large_prob += xl[f] <= BIG_M_L * yl[f]

    
    large_prob += lp.lpSum([yl[f] for f in valid_foods]) >= 5

    large_prob.solve(lp.PULP_CBC_CMD(msg=False))

    print("\n--- Large diet model (minimize cholesterol) ---")
    print("Cholesterol column used:", _chol_col)
    print("Status:", lp.LpStatus[large_prob.status])
    print("Total cholesterol:", round(lp.value(large_prob.objective), 4))
    shown = 0
    for f in valid_foods:
        serv = xl[f].value()
        chosen = yl[f].value()
        if chosen is not None and int(chosen) == 1 and serv is not None and serv > 1e-5:
            print(f"{f}: {serv:.3f}")
            shown += 1
            if shown >= 15:
                break