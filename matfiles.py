import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ------------------------------
# 🔧 Function to read MATLAB v7.3+ (.mat) file
# ------------------------------
def read_mat_v73(filepath):
    data = {}
    with h5py.File(filepath, 'r') as f:
        def recursively_load(name, obj):
            if isinstance(obj, h5py.Dataset):
                try:
                    data[name] = obj[()]
                except Exception as e:
                    data[name] = f"Error reading: {e}"
        f.visititems(recursively_load)
    return data

# ------------------------------
# 🧾 Function to summarize contents
# ------------------------------
def summarize_data(data):
    summary = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            summary[k] = f"ndarray, shape={v.shape}, dtype={v.dtype}"
        else:
            summary[k] = str(v)
    return summary

# ------------------------------
# 📂 File paths (update if needed)
# ------------------------------
file1 = '/mnt/DATA/Vamsi/EE24M309/code_final/NSDeblur/ICK/ICK_40.mat'
file2 = '/mnt/DATA/Vamsi/EE24M309/code_final/ICK_20.mat'

# ------------------------------
# ✅ Ensure both files exist
# ------------------------------
assert os.path.exists(file1), f"File not found: {file1}"
assert os.path.exists(file2), f"File not found: {file2}"

# ------------------------------
# 📥 Load both .mat files
# ------------------------------
data1 = read_mat_v73(file1)
data2 = read_mat_v73(file2)

# ------------------------------
# 🧾 Summarize structure of each file
# ------------------------------
summary1 = summarize_data(data1)
summary2 = summarize_data(data2)

df1 = pd.DataFrame.from_dict(summary1, orient='index', columns=['File 1'])
df2 = pd.DataFrame.from_dict(summary2, orient='index', columns=['File 2'])

comparison = pd.concat([df1, df2], axis=1)
comparison.index.name = 'Variable Name'

print("\n📊 Summary of Variables:\n")
print(comparison)

# ------------------------------
# 🔍 Compare actual values of the 'Kernel' variable
# ---------------------
