# coding=utf-8

# the first spark app
from pyspark import SparkContext

# create SparkContext
sc=SparkContext("local[2]", "First Spark App")

# get data source
data=sc.textFile("/user-program/python/MachineLearningSpark/Data/UserPurchaseHistory.txt").map(lambda
line:line.split(",")).map(lambda record:(record[0], record[1], record[2]))

# the number of purchases
numPurchases=data.count()

# the unique user
uniqueUser=data.map(lambda record:record[0]).distinct().count()

# the total revenue
totalRevenue=data.map(lambda record:float(record[2])).sum()

#the most popular product
products=data.map(lambda record:(record[1], 1)).reduceByKey(lambda a, b:a+b).collect()

sc.stop()

print("the number of purchases :"+str(numPurchases))
print(uniqueUser)
print(totalRevenue)
print(products)
