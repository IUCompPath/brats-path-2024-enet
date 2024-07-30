import os
import random
import pickle

root_dir = "./dataset"
classes = os.listdir(root_dir)

all_fold_paths = {1:[], 2:[], 3:[], 4:[], 5:[]}

for cls in classes:
    class_dir = os.path.join(root_dir, cls)
    images = os.listdir(class_dir)
    images = [f'{class_dir}/' + image for image in images]
    random.shuffle(images)
    num = len(images) // 5
    im1 = images[:num]
    all_fold_paths[1].extend(im1)
    im2 = images[num:2*num]
    all_fold_paths[2].extend(im2)
    im3 = images[2*num:3*num]
    all_fold_paths[3].extend(im3)
    im4 = images[3*num:4*num]
    all_fold_paths[4].extend(im4)
    im5 = images[4*num:]
    all_fold_paths[5].extend(im5)

with open('image_paths/all_fold_paths.pkl', 'wb') as file:
    pickle.dump(all_fold_paths, file)