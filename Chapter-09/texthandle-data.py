# coding=utf-8

import re
import string
import unicodedata
import sys
from pyspark import SparkContext, SparkConf

#
conf = SparkConf().setAppName("Text Handle Data").setMaster("local[4]")
sc= SparkContext(conf=conf)

#
rdd = sc.wholeTextFiles("/user-program/python/MachineLearningSpark/Data/20news-bydate/20news-bydate-train/*")
text = rdd.map(lambda (file, text) : text)
print("the total number of 20news by date train :" + str(text.count()))

#
newsgroups = rdd.map(lambda (file, text) : file.split('/')[-2])
print(newsgroups.first())
count_group = newsgroups.map(lambda l : (l, 1)).reduceByKey(lambda a, b : a+b).collect()
print(count_group)

#
def show_list(ls):
    for l in ls:
        print(l)

tbl = dict.fromkeys(i for i in xrange(sys.maxunicode)
                    if unicodedata.category(unichr(i)).startswith('P'))
def remove_punctuation(text):
    return text.translate(tbl)

remove_punct_map = dict.fromkeys(map(ord,string.punctuation))
#
white_space_split = text.flatMap(lambda t : t.split(" ")).map(lambda w : w.lower())
print("---------------white space split :" + str(white_space_split.distinct().count()))
show_list(white_space_split.distinct().sample(True, 0.3, 42).take(100))

no_word_split = text.map(lambda w : w.lower()).map(lambda w : w.translate(remove_punct_map))\
    .filter(lambda w : w != None).flatMap(lambda t : t.split(" "))
print("---------------no word split :" + str(no_word_split.distinct().count()))
show_list(no_word_split.distinct().sample(True, 0.3, 42).take(100))

filter_numbers = no_word_split.filter(lambda w : re.match(r'[^0-9]*', w).group())
print("---------------filter numbers :" + str(filter_numbers.distinct().count()))
show_list(filter_numbers.distinct().sample(True, 0.3, 42).take(100))

# token_counts = filter_numbers.map(lambda w : (w, 1)).reduceByKey(lambda a, b : a+b).collect()
# print(token_counts)
#
stop_words = ["the","a","an","of","or","in","for","by","on","but", "is", "not",
              "with", "as", "was", "if","they", "are", "this", "and", "it", "have",
              "from", "at", "my","be", "that", "to"]

sc.stop()