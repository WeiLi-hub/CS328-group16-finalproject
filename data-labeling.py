
"""
This is the script used to combine all collected csv data files into
a single csv file.
"""

import numpy as np
import csv
import time
import os

import labels


# print the available class label (see labels.py)
act_labels = labels.activity_labels
print(act_labels)

# specify the data files and corresponding activity label
csv_directory = ["data/backhand", "data/overhead", "data/none"]
activity_list = ["backhand", "overhead", "None"]
# Specify final output file name. 
output_filename = "data/all_labeled_data.csv"
output_filename_gyro = "data/all_labeled_data_gyro.csv"


all_data = []
gyro = []

zip_list = zip(csv_directory, activity_list)

for directory, act in zip_list:
    print(directory)
    print(act)
    if act in act_labels:
        label_id = act_labels.index(act)
    else:
        print("Label: " + act + " NOT in the activity label list! Check label.py")
        exit()

    f = []
    for file in os.listdir(directory):
        f.append(file)
    f = sorted(f)
    
    for file in f:
        print("Process file: " + file + " and assign label: " + act + " with label id: " + str(label_id))

        with open(directory +  '/' + file, "r") as f:
            if 'gyro' in file.lower():
                reader = csv.reader(f, delimiter = ",")
                headings = next(reader)
                for row in reader:
                    row.append(str(label_id))
                    gyro.append(row)

            else:
                reader = csv.reader(f, delimiter = ",")
                headings = next(reader)
                for row in reader:
                    row.append(str(label_id))
                    all_data.append(row)

# print(len(all_data))
# print(len(gyro))

# make gyro and acceleration data the same length
# reason we make the same length is because missing some data or
# human error when preprocessing data
m = min(len(all_data), len(gyro))
all_data = all_data[:m]
gyro = gyro[:m]

with open(output_filename, 'w',  newline='') as f:
    writer = csv.writer(f)
    writer.writerows(all_data)
    print("Data saved to: " + output_filename)

with open(output_filename_gyro, 'w',  newline='') as f:
    writer = csv.writer(f)
    writer.writerows(gyro)
    print("Data saved to: " + output_filename_gyro)
