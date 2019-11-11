import csv
import matplotlib.pyplot as plt
import statistics
import numpy as np

data = []
with open('Dataset.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            temp = [float(row[0]),float(row[1])]
            data.append(temp)
            line_count += 1

plt.scatter(np.array(data)[:,0], np.array(data)[:,1], s=10)
plt.show()