import sys
import numpy as np

arg1 = sys.argv[1]

data = open(arg1, "r").read().split("\n")

cm = np.zeros((2,2))
label_data = ['yes', 'no']


for val in data:
    if val  == "" or val == " ":
        continue
    cols = val.split("\t")
    label = cols[2]
    pred = cols[4]
    label_index = label_data.index(label)
    pred_index = label_data.index(pred)

    cm[label_index][pred_index] += 1

print(cm)
