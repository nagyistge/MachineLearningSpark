# coding=utf-8

from pyspark import SparkContext, SparkConf
from pyspark.mllib.recommendation import ALS
from pyspark.mllib.recommendation import Rating
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.clustering import KMeans
import numpy as np

conf = SparkConf().setAppName("Cluster KMeans").setMaster("local[2]")
sc = SparkContext(conf=conf)

#
row_data = sc.textFile("/user-program/python/MachineLearningSpark/Data/ml-100k/u.data")
row_ratings = row_data.map(lambda line : line.split('\t')).map(lambda r : Rating(int(r[0]), int(r[1]), float(r[2])))
print(row_ratings.first())

#
row_ratings.cache()

#
als_model = ALS.train(row_ratings, 50, 10, 0.1)
movie_factors = als_model.productFeatures().map(lambda (id, factor) : (id, Vectors.dense(factor)))
movie_vectors = movie_factors.map(lambda (id, vector): vector)
#print(movie_vectors.first())
user_factors = als_model.userFeatures().map(lambda (id, factor) : (id, Vectors.dense(factor)))
user_vectors = user_factors.map((lambda (id, vector) : vector))
#print(user_vectors.first())

# train
movie_cluster_model = KMeans().train(movie_vectors, k=5, maxIterations=10, runs=3)
print("movie cluster model kmeans :")
print(movie_cluster_model)
user_cluster_model = KMeans().train(user_vectors, k=5, maxIterations=10, runs=3)
print("user cluster model kmeans :")
print(user_cluster_model)

# predict
movie_1 = movie_vectors.first()
movie_cluster = movie_cluster_model.predict(movie_1)
print(movie_cluster)

# evaluation
movie_cost = movie_cluster_model.computeCost(movie_vectors)
print("WCSS for movies :" + str(movie_cost))
train_test_split_movies = movie_vectors.randomSplit((0.6, 0.4), 123)
train_movies = train_test_split_movies[0]
test_movies = train_test_split_movies[1]
def costs_movies(cluster, train, test):
    for c in cluster:
        m = KMeans().train(train, k=c, maxIterations=10, runs=3)
        wscc = m.computeCost(test)
        print("WSCC for k=" + str(c) + ":" + str(wscc))
cluster_list = [2, 3, 4, 5, 10, 20]
costs_movies(cluster_list, train_movies, test_movies)


sc.stop()