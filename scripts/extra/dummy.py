# import pickle

# imgs_path_file = './image_paths/all_fold_paths.pkl'

# data = []

# with open(imgs_path_file, 'rb') as fp:
#     all_fold_paths = pickle.load(fp)

# all_folds = [1, 2, 3, 4, 5]
# selec_folds = [x for x in all_folds if x != 3]

# for selec_fold in selec_folds:
#     paths = all_fold_paths[selec_fold]

#     for path in paths:
#         class_name = path.split("_")[-2]
#         data.append((path, class_name))
#     print(data)
#     break

import yaml

parameters = {
    'batch_size': 32,
    'image_size': [512,512],
}

yaml_file_path = './ml-cube-brats/mlcube/workspace/parameters.yaml'

# Step 3: Save the dictionary to a YAML file
with open(yaml_file_path, 'w') as file:
    yaml.dump(parameters, file, default_flow_style=False)