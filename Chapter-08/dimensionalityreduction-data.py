# coding=utf-8

from pyspark import SparkContext, SparkConf
import matplotlib as mp

conf = SparkConf().setAppName("Dimensionality Reduction Data").setMaster("local[2]")
sc = SparkContext(conf=conf)

#
rdd = sc.wholeTextFiles("/user-program/python/MachineLearningSpark/Data/lfw/*")
first = rdd.first()
print(first)
files = rdd.map(lambda (file_name, content) : file_name.replace("file:", ""))
print(files.first())
print("the number of files :" + str(files.count()))


sc.stop()