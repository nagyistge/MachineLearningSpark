# coding=utf-8

#movie stream
from pyspark import SparkContext, SparkConf

conf = SparkConf().setAppName("User Information").setMaster("local[2]")
sc = SparkContext(conf = conf)
#sc=SparkContext("local[2]", "MovieStream")

user_data = sc.textFile("/user-program/python/MachineLearningSpark/Data/ml-100k/u.user")
first_data = user_data.first()
take_data = user_data.take(2)

user_fields = user_data.map(lambda line : line.split("|"))

#the number of user
num_users = user_fields.map(lambda fields : fields[0]).count()
#the number of gender
num_genders = user_fields.map(lambda fields : fields[2]).distinct().count()
#the number of occupation
num_occupations = user_fields.map(lambda fields : fields[3]).distinct().count()
#the number of zipcode
num_zipcode = user_fields.map(lambda fields : fields[4]).distinct().count()

#account occupation
account_occupation = user_fields.map(lambda fields : (fields[3], 1)).reduceByKey(lambda x, y : x+y).collect()
account_occupation2 = user_fields.map(lambda fields : fields[3]).countByValue()

sc.stop()

#show the information must by print
print("u.user first data :" + first_data)
print("u.user take two :" + take_data[0] +"--"+take_data[1])
print("users:" + str(num_users))
print("genders:" + str(num_genders))
print("occupations:" + str(num_occupations))
print("zipcode:" + str(num_zipcode))
print("different occupation percentage:")
print(account_occupation)
print("different occupation percentage by 'count by value':")
print(account_occupation2)
