import csv
import os
import pandas as pd

# directory_redirect
current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
grandparent_directory = os.path.dirname(parent_directory)
grandparent_directory_new = os.path.dirname(grandparent_directory)
print(grandparent_directory_new)
file_path = os.path.join(grandparent_directory_new, 'data/ukbb/ukb668815_imaging.csv')

# 读取初始列和要添加的列范围
initial_cols = [1, 977, 978, 979, 980, 6603,10362,10229,10377]
additional_cols_range = range(3325, 3337)
additional_diabets = range(1033, 1037)
add=range(10357,10362)
Treatment=range(4218,4410)
cols_to_read = initial_cols + list(additional_cols_range)+list(additional_diabets)+list(add)+list(Treatment)
df = pd.read_csv(file_path, usecols=cols_to_read, low_memory=False)

print(df.head(35))
output_file_path = '/home/ruochen/projects/kmae/results.csv'

df.to_csv(output_file_path, index=False)
