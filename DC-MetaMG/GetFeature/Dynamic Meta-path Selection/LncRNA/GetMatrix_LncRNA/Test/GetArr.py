import numpy as np

# 定义一个二维数组
array = np.zeros([60, 1937], dtype=int)


Drug_dict = {}
Disease_dict = {}
# 读取文件内容并存储到字典中
with open('../drug.txt', 'r') as file:
    for line in file:
        # 去掉每行的换行符并按制表符分割
        line = line.strip()
        if line:  # 确保不是空行
            key, value = line.split('\t')
            Drug_dict[value] = int(key)  # 以 miRNA 名称为键，序号为值

with open('../Disease.txt', 'r') as file:
    for line in file:
        # 去掉每行的换行符并按制表符分割
        line = line.strip()
        if line:  # 确保不是空行
            key, value = line.split('\t')
            Disease_dict[value] = int(key)  # 以 miRNA 名称为键，序号为值

with open('../DDI.txt', 'r') as file:
    for line in file:
        line = line.strip().split('\t')
        array[int(Drug_dict[line[0]]),int(Disease_dict[line[1]])]=1
# 将数组写入文本文件
with open('../DD_ARR.txt', 'w') as file:
    for row in array:
        file.write(' '.join(map(str, row)) + '\n')  # 每行的所有元素用空格分隔