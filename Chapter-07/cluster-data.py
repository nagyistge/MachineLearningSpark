# coding=utf-8

from pyspark import SparkContext, SparkConf
import numpy as np

conf = SparkConf().setAppName("Cluster Data").setMaster("local[2]")
sc = SparkContext(conf=conf)

movies = sc.textFile("/user-program/python/MachineLearningSpark/Data/ml-100k/u.item")
print("the first data of movies :" + str(movies.first()))

#
genres = sc.textFile("/user-program/python/MachineLearningSpark/Data/ml-100k/u.genre")
print(genres.take(5))
genre_map = genres.filter(lambda g : g != '') .map(lambda line : line.split("|")).map(lambda l : (l[1], l[0])).collectAsMap()
print(genre_map)

#
#def get_genres
def get_titles_genres(array, genre_map):
    genres = array[5 : len(array)]
    genres_assigned = []
    for i in range(len(genres)):
        if genres[i] == "1":
            genres_assigned.append(genre_map.get(str(i)))
        else:
            pass
    return (int(array[0]), (array[1], tuple(genres_assigned)))

titles_genres = movies.map(lambda line : line.split("|")).map(lambda array : get_titles_genres(array, genre_map))

print(titles_genres.first())


sc.stop()