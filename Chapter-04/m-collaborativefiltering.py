# coding=utf-8

from pyspark import SparkContext, SparkConf
from pyspark.mllib.recommendation import ALS
from pyspark.mllib.recommendation import Rating
import numpy


conf = SparkConf().setAppName("Recommend Movie").setMaster("local[2]")
sc = SparkContext(conf = conf)

#get the original data from u.data
data_rates = sc.textFile("/user-program/python/MachineLearningSpark/Data/ml-100k/u.data")
data_movies = sc.textFile("/user-program/python/MachineLearningSpark/Data/ml-100k/u.item")
first_rate = data_rates.first()
print("the first data from rate :" + first_rate)

#get the fields user_id, movie_id and movie_rate
raw_rates = data_rates.map(lambda line : line.split("\t")).map(lambda line : (line[0], line[1], line[2]))
print("the first data from raw_data :")
print(raw_rates.first())

#transform data get Rating format data
rates = raw_rates.map(lambda t : Rating(int(t[0]), int(t[1]), float(t[2])))
print("the first data of Rating format :")
print(rates.first())

#train
model = ALS.train(rates, 50, 10, 0.01)
user_features = model.userFeatures()
product_features = model.productFeatures()

print("the number of user features :" + str(user_features.count()))
print("the number of product features :" + str(product_features.count()))

#predict user id '789' product id '123'
predict_rate = model.predict(789, 123)
print("the predict rate :" + str(predict_rate))

#recommend products for user id 789 num 10, top 10
recommend_products = model.recommendProducts(789, 10)
print("top 10 products to user id 789 :")
print(recommend_products)

#verify
titles = data_movies.map(lambda line : line.split("|")).map(lambda line : (int(line[0]), line[1])).collectAsMap()
print("title 123 :"+ str(titles[123]))
user_movies = rates.filter(lambda rate : rate.user==789)
print("user 789 rates movies num :")
print(user_movies.count())
user_movies_top = user_movies.sortBy(lambda rate : rate.rating, ascending=False)\
    .map(lambda rate : (titles[rate.product], rate.rating)).take(10)
print(user_movies_top)

for recommend_product in recommend_products:
    recommend_movie = (titles[recommend_product.product], recommend_product.rating)
    print(recommend_movie)

sc.stop()