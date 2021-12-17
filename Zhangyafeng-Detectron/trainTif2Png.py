import os
import shutil
from tqdm import tqdm
import cv2

images_dir = '/tmp/pycharm_project_957/sartorius-cell-instance-segmentation/LIVECell_dataset_2021/images/livecell_test_images'
output_dir = '/tmp/pycharm_project_957/sartorius-cell-instance-segmentation/LIVECell_dataset_2021/images/new_livecell_test_images'

names = os.listdir(images_dir)
for i in range(len(names)):
    save_images_dir = os.path.join(output_dir, names[i])
    if not os.path.exists(save_images_dir): os.makedirs(save_images_dir)

for i in range(len(names)):
    name_list = os.listdir(os.path.join(images_dir, names[i]))
    images_dir_path = os.path.join(images_dir, names[i])

    for j in range(len(name_list)):
        tif_path = os.path.join(images_dir_path, name_list[j])
        tif_bgr = cv2.imread(tif_path)
        path = os.path.join(os.path.join(output_dir, names[i]),
                     name_list[j].replace(".tif", ".png"))
        cv2.imwrite(path, tif_bgr)



