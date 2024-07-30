import pickle

imgs_path_file = './image_paths/all_fold_paths.pkl'

data = []

with open(imgs_path_file, 'rb') as fp:
    all_fold_paths = pickle.load(fp)

all_folds = [1, 2, 3, 4, 5]
selec_folds = [x for x in all_folds if x != 3]

for selec_fold in selec_folds:
    paths = all_fold_paths[selec_fold]

    for path in paths:
        class_name = path.split("_")[-2]
        data.append((path, class_name))
    print(data)
    break