import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os

def get_clean_dataframe():
    path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")

    metadata_path = os.path.join(path,'HAM10000_metadata.csv')

    df = pd.read_csv(metadata_path)

    def get_path(img_id):
        p1 = os.path.join(path, "HAM10000_images_part_1",img_id + '.jpg')
        p2 = os.path.join(path,'HAM10000_images_part_2',img_id + '.jpg')
        return p1 if os.path.exists(p1) else p2

    df['path'] = df['image_id'].apply(get_path)
    return df
 




