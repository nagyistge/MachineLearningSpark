# coding=utf-8

from pyspark import SparkContext, SparkConf

conf = SparkConf().setAppName("Rate information").setMaster("local[2]")
sc = SparkContext(conf = conf)

#get rate data
rate_data = sc.textFile("/user-program/python/MachineLearningSpark/Data/ml-100k/u.data")
#get u.data first data
first_data = rate_data.first()
#get the number of rates
num_rates = rate_data.count()
#get rate fields
rate_fields = rate_data.map(lambda line : line.split("\t"))
#get the number of rate by user mark
user_rates_group = rate_fields.map(lambda fields : (fields[0], fields[2])).groupByKey().map(lambda (x, y) : (x, len(y)))
num_group = user_rates_group.count()
user_rates_take = user_rates_group.take(5)


sc.stop()

print("the u.data first data :" + str(first_data))
print("the number of rates :" + str(num_rates))
print("the number of rate by user group :" + str(num_group))
print("the number of rate by user mark :")
print(user_rates_take)