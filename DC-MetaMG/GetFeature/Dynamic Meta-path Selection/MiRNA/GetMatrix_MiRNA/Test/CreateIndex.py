import openpyxl
import pandas as pd
import numpy as np
mirnas = []
mirnas1 = []

with open("../Disease.txt", "r") as infile:
    for line in infile:
        mirnas.append(line.strip())


a = 0
with open('Disease1.txt', 'w') as infile:
    for i in mirnas:
        infile.write(str(a)+'\t'+i+'\n')
        a+=1
