# coding=utf-8

from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint

import numpy as np
import matplotlib.pyplot as plt


conf = SparkConf().setAppName("extract features").setMaster("local[2]")
sc = SparkContext(conf = conf)

raw_data = sc.textFile("/user-program/python/MachineLearningSpark/Data/bike-sharing/hour-noheader.csv")
num_data = raw_data.count()
records = raw_data.map(lambda line : line.split(','))
first = records.first()
print("the number of hour-noheader.csv :" + str(num_data))
print("the first data of records :")
print(first)

records.cache()

#
def get_mapping(rdd, idx):
    return rdd.map(lambda line : line[idx]).distinct().zipWithIndex().collectAsMap()

print("column 3 index 2 :")
print(get_mapping(records, 2))

mappings = [get_mapping(records, i) for i in range(2, 10)]
cat_len = sum(map(len, mappings))
num_len = len(records.first()[11 : 15])
total_len = cat_len  + num_len
print("total feature vector length :" + str(total_len))

#
# def extract_features(record):
#     cat_vec = np.zeros(cat_len)
#     i = 0
#     step = 0
#     for field in record[2 : 9]:
#         m = mappings[i]
#         idx = m[field]
#         cat_vec[idx + step] = 1
#         i = i + 1
#         step = step + len(m)
#     num_vec = np.array([float(field) for field in record[10 : 14]])
#     return np.concatenate((cat_vec, num_vec))
#
# def extract_label(record):
#     return float(record[-1])

#
# target = records.map(lambda r : r[-1]).collect()
# plt.hist(target, bins=40, color="lightblue", normed=True)
# fig = plt.gcf()
# fig.set_size_inches(16, 10)

sc.stop()