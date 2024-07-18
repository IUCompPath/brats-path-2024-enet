import os
import numpy as np
import pickle
import pandas as pd

ms_sheet = pd.read_csv("updated_master_sheet.csv")

pats = [f"Pat{x}" for x in range(1,85)]

np.random.shuffle(pats)

test = pats[:8]
f0 = pats[8:15]
f1 = pats[15:22]
f2 = pats[22:29]
f3 = pats[29:37]
f4 = pats[37:45]
f5 = pats[45:53]
f6 = pats[53:61]
f7 = pats[61:69]
f8 = pats[69:77]
f9 = pats[77:]

all_fold_paths = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
test_paths = []

for i, pat in enumerate(ms_sheet.iloc[:,1]):
    if pat in test:
        test_paths.append(ms_sheet.iloc[i,2])
    elif pat in f0:
        all_fold_paths[0].append(ms_sheet.iloc[i,2])
    elif pat in f1:
        all_fold_paths[1].append(ms_sheet.iloc[i,2])
    elif pat in f2:
        all_fold_paths[2].append(ms_sheet.iloc[i,2])
    elif pat in f3:
        all_fold_paths[3].append(ms_sheet.iloc[i,2])
    elif pat in f4:
        all_fold_paths[4].append(ms_sheet.iloc[i,2])
    elif pat in f5:
        all_fold_paths[5].append(ms_sheet.iloc[i,2])
    elif pat in f6:
        all_fold_paths[6].append(ms_sheet.iloc[i,2])
    elif pat in f7:
        all_fold_paths[7].append(ms_sheet.iloc[i,2])
    elif pat in f8:
        all_fold_paths[8].append(ms_sheet.iloc[i,2])
    elif pat in f9:
        all_fold_paths[9].append(ms_sheet.iloc[i,2])

# classes = os.listdir("./dataset/")

# for classname in classes:
#     class_files = os.listdir(f"./dataset/{classname}")
#     np.random.shuffle(class_files)
#     count_per_fold = len(class_files) // 11
#     test_paths.extend(class_files[:count_per_fold])
#     all_fold_paths[0].extend(class_files[count_per_fold:2*count_per_fold])
#     all_fold_paths[1].extend(class_files[2*count_per_fold:3*count_per_fold])
#     all_fold_paths[2].extend(class_files[3*count_per_fold:4*count_per_fold])
#     all_fold_paths[3].extend(class_files[4*count_per_fold:5*count_per_fold])
#     all_fold_paths[4].extend(class_files[5*count_per_fold:6*count_per_fold])
#     all_fold_paths[5].extend(class_files[6*count_per_fold:7*count_per_fold])
#     all_fold_paths[6].extend(class_files[7*count_per_fold:8*count_per_fold])
#     all_fold_paths[7].extend(class_files[8*count_per_fold:9*count_per_fold])
#     all_fold_paths[8].extend(class_files[9*count_per_fold:10*count_per_fold])
#     all_fold_paths[9].extend(class_files[10*count_per_fold:])

for key in all_fold_paths:
    print(len(all_fold_paths[key]))

with open('all_test_paths.pkl', 'wb') as fp:
    pickle.dump(test_paths, fp)
    print('List saved successfully to file')

with open('all_fold_paths.pkl', 'wb') as fp:
    pickle.dump(all_fold_paths, fp)
    print('dictionary saved successfully to file')

with open('all_test_paths.pkl', 'rb') as fp:
    loaded_test_paths = pickle.load(fp)