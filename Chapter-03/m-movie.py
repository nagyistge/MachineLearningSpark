# coding=utf-8

def convert_year(x):
    try:
        return int(x[-4:])
    except:
        return 1900


from pyspark import SparkContext, SparkConf

conf = SparkConf().setAppName("Movie Information").setMaster("local[2]")
sc = SparkContext(conf = conf)

movie_data = sc.textFile("/user-program/python/MachineLearningSpark/Data/ml-100k/u.item")

#get the first data
first_data = movie_data.first()
#get movie fields, the data type is list
movie_fields = movie_data.map(lambda line : line.split("|"))
#account movie age
years = movie_fields.map(lambda fields : fields[2]).map(lambda year : convert_year(year))
years_filtered = years.filter(lambda year : year !=1900)
movie_ages = years_filtered.map(lambda year : 1998 - year).countByValue()

sc.stop()

print("the first data of movie :" + str(first_data))
print("the movie ages :")
print(movie_ages)