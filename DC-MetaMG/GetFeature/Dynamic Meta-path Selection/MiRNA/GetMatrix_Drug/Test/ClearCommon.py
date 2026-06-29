import openpyxl
import pandas as pd
import numpy as np
mirnas = []
mirnas1 = []

with open("../Disease.txt", "r") as infile:
    for line in infile:
        mirnas.append(line.strip())



common_elements = np.intersect1d(mirnas, mirnas)

with open('Disease1.txt', 'w') as infile:
    for i in common_elements:
        infile.write(i+'\n')
# 计算相同元素的数量
num_common_elements = len(common_elements)

