# coding=utf-8
print ("This file is about spark document !")

#if __name__ == '__main__':

from pyspark import SparkContext, SparkConf

#1 create SparkContext object
conf = SparkConf().setAppName("SparkTest").setMaster("local[2]")
sc = SparkContext(conf = conf)

#2 create RDD object
#2.1 way one
rdd1 = sc.textFile("/user-program/python/MachineLearningSpark/Data/UserPurchaseHistory.txt")
#2.2 way two parallelize
data = [1, 2, 3, 4, 5, 6]
rdd2 = sc.parallelize(data)

#3 RDD has two operations, one operation is transform, another operation is action
rdd = sc.textFile("/user-program/python/MachineLearningSpark/Data/UserPurchaseHistory.txt")
dist_file = rdd.flatMap(lambda line : line.split(",")).map(lambda line : len(line)).reduce(lambda x, y : x + y)
print("the file length :" + str(dist_file))

#4 data persist or cache
data_persist = rdd.flatMap(lambda line : line.split(",")).persist()

#5 key value
word_count = rdd.flatMap(lambda line : line.split(",")).map(lambda word : (word, 1)).reduceByKey(lambda x, y : x + y)\
.collect()
print(word_count)
word_count_sort = rdd.flatMap(lambda line : line.split(",")).map(lambda word : (word, 1)).reduceByKey(lambda x, y : x + y)\
.sortByKey().collect()
print(word_count_sort)

sc.stop()