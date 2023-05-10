
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
# csv_files = ["data/running.csv", "data/sitting.csv", "data/stairs.csv", "data/stairs2.csv", "data/walking.csv"]
# activity_list = ["running", "sitting", "stairs_down","stairs_down", "walking"]

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

    for file in os.listdir(directory):
        print("Process file: " + file + " and assign label: " + act + " with label id: " + str(label_id))
        with open(directory +  '/' + file, "r") as f:
            reader = csv.reader(f, delimiter = ",")
            headings = next(reader)
            for row in reader:
                row.append(str(label_id))
                if file.find('gyro') != -1:
                    gyro.append(row)
                else:
                    all_data.append(row)


with open(output_filename, 'w',  newline='') as f:
    writer = csv.writer(f)
    writer.writerows(all_data)
    print("Data saved to: " + output_filename)

with open(output_filename_gyro, 'w',  newline='') as f:
    writer = csv.writer(f)
    writer.writerows(gyro)
    print("Data saved to: " + output_filename_gyro)



