# coding=utf-8

from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
import numpy as np

conf = SparkConf().setAppName("Regression Decision Tree Model").setMaster("local[2]")
sc = SparkContext(conf = conf)

raw_data = sc.textFile("/user-program/python/MachineLearningSpark/Data/bike-sharing/hour-noheader.csv")
num_data = raw_data.count()
records = raw_data.map(lambda line : line.split(','))
first = records.first()
print("the number of hour-noheader.csv :" + str(num_data))
print("the first data of records :")
print(first)

records.cache()

#
def extract_features(record):
    return np.array(map(float, record[2 : 14]))

def extract_label(record):
    return float(record[-1])

data = records.map(lambda r : LabeledPoint(extract_label(r), extract_features(r)))
first_point = data.first()
print("Raw data :")
print(first[2 : ])
print("Label")
print(first_point.label)
print("decision tree model feature vector :")
print(first_point.features)
print("decision tree model feature vector length :" + str(len(first_point.features)))

#
dt_model = DecisionTree.trainRegressor(data, {})
preds = dt_model.predict(data.map(lambda d : d.features))
actual = data.map(lambda d : d.label)
true_vs_predicted = actual.zip(preds)
print("decision tree prediction :" + str(true_vs_predicted.take(5)))
print("decision tree depth :" + str(dt_model.depth()))
print("decision tree number of nodes :" + str(dt_model.numNodes()))

#
def squared_error(actual, prediction):
    return (actual - prediction)**2

def abs_error(actual, prediction):
    return np.abs(actual - prediction)

def squared_log_error(actual, prediction):
    return (np.log(prediction + 1) - np.log(actual + 1))**2

mse = true_vs_predicted.map(lambda (a, p) : squared_error(a, p)).mean()
msa = true_vs_predicted.map(lambda (a, p) : abs_error(a, p)).mean()
rmsle = true_vs_predicted.map(lambda (a, p) : squared_log_error(a, p)).mean()

print("decision tree model-squared error :" + str(mse))
print("decision tree model-absolute error :" + str(msa))
print("decision tree model-root mean squared log error :" + str(rmsle))

#
data_dt_log = data.map(lambda lp : LabeledPoint(np.log(lp.label), lp.features))
dt_model_log = DecisionTree.trainRegressor(data_dt_log, {})
preds_log = dt_model_log.predict(data_dt_log.map(lambda lp : lp.features))
actual_log = data_dt_log.map(lambda lp : lp.label)
true_vs_predicted_log = actual_log.zip(preds_log).map(lambda (t, p) : (np.exp(t), np.exp(p)))

mse_log = true_vs_predicted_log.map(lambda (a, p) : squared_error(a, p)).mean()
msa_log = true_vs_predicted_log.map(lambda (a, p) : abs_error(a, p)).mean()
rmsle_log = true_vs_predicted_log.map(lambda (a, p) : squared_log_error(a, p)).mean()
print("decision tree model log squared error :" + str(mse_log))
print("decision tree model log absolute error :" + str(msa_log))
print("decision tree model log root mean squared log error :" + str(rmsle_log))

sc.stop()