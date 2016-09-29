# coding=utf-8

from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.regression import LinearRegressionWithSGD
import numpy as np

conf = SparkConf().setAppName("Regression Linear Model").setMaster("local[2]")
sc = SparkContext(conf = conf)

raw_data = sc.textFile("/user-program/python/MachineLearningSpark/Data/bike-sharing/hour-noheader.csv")
num_data = raw_data.count()
records = raw_data.map(lambda line : line.split(','))
first = records.first()
print("the number of hour-noheader.csv :" + str(num_data))
print("the first data of records :")
print(first)

#
records.cache()

#
def get_mapping(rdd, idx):
    return rdd.map(lambda line : line[idx]).distinct().zipWithIndex().collectAsMap()

print("column 3 index 2 :")
print(get_mapping(records, 2))

mappings = [get_mapping(records, i) for i in range(2, 10)]
cat_len = sum(map(len, mappings))
num_len = len(records.first()[11 : 15])
total_len = cat_len  + num_len
print("total feature vector length :" + str(total_len))

#
def extract_features(record):
    cat_vec = np.zeros(cat_len)
    i = 0
    step = 0
    for field in record[2 : 9]:
        m = mappings[i]
        idx = m[field]
        cat_vec[idx + step] = 1
        i = i + 1
        step = step + len(m)
    num_vec = np.array([float(field) for field in record[10 : 14]])
    return np.concatenate((cat_vec, num_vec))

def extract_label(record):
    return float(record[-1])

data = records.map(lambda r : LabeledPoint(extract_label(r), extract_features(r)))
first_point = data.first()
print("Raw data :")
print(first[2 : ])
print("Label")
print(first_point.label)
print("Linear model feature vector :")
print(first_point.features)
print("Linear model feature vector length :" + str(len(first_point.features)))

#
linear_model = LinearRegressionWithSGD.train(data, iterations=10, step=0.1, intercept=False)
true_vs_predicted = data.map(lambda p : (p.label, linear_model.predict(p.features)))
print("linear model predictions :" + str(true_vs_predicted.take(5)))

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

print("linear model-squared error :" + str(mse))
print("linear model-absolute error :" + str(msa))
print("linear model-root mean squared log error :" + str(rmsle))

#
data_log = data.map(lambda lp : LabeledPoint(np.log(lp.label), lp.features))
model_log = LinearRegressionWithSGD.train(data_log, iterations=10, step=0.1)
true_vs_predicted_log = data_log.map(lambda p :(np.exp(p.label), np.exp(model_log.predict(p.features))))

mse_log = true_vs_predicted_log.map(lambda (a, p) : squared_error(a, p)).mean()
msa_log = true_vs_predicted_log.map(lambda (a, p) : abs_error(a, p)).mean()
rmsle_log = true_vs_predicted_log.map(lambda (a, p) : squared_log_error(a, p)).mean()
print("linear model log squared error :" + str(mse_log))
print("linear model log absolute error :" + str(msa_log))
print("linear model log root mean squared log error :" + str(rmsle_log))

#
data_with_idx  = data.zipWithIndex().map(lambda (k, v) : (v, k))
test = data_with_idx.sample(False, 0.2, 42)
train = data_with_idx.subtractByKey(test)

train_data = train.map(lambda (k, v) : v)
test_data = test.map(lambda (k, v) : v)
train_size = train_data.count()
test_size = test_data.count()

print("train data size :" + str(train_size))
print("test data size :" + str(test_size))
print("total data size :" + str(num_data))
print("train and test data size :" + str(train_size + test_size))

#
def evaluate(train, test, iterations, step, regParam, regType, intercept):
    model = LinearRegressionWithSGD.train(train, iterations, step, regParam=regParam, regType=regType, intercept=intercept)
    tp = test.map(lambda p: (p.label, model.predict(p.features)))
    rmsle = np.sqrt(tp.map(lambda (t, p): squared_log_error(t, p)). mean())
    return rmsle

params_iteration = [1, 5, 10, 20, 50, 100]
metrics_iteration = [evaluate(train_data, test_data, param, 0.01, 0.0, 'l2', False) for param in params_iteration]
print(params_iteration)
print(metrics_iteration)

params_step = [0.01, 0.025, 0.05, 0.1, 1.0]
metrics_step = [evaluate(train_data, test_data, 10, param, 0.0, 'l2', False) for param in params_step]
print(params_step)
print(metrics_step)

params_reg_2 = [0.0, 0.01, 0.1, 1.0, 5.0, 10.0, 20.0]
metrics_reg_2 = [evaluate(train_data, test_data, 10, 0.1, param, 'l2', False) for param in params_reg_2]
print(params_reg_2)
print(metrics_reg_2)

params_reg_1 = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
metrics_reg_1 = [evaluate(train_data, test_data, 10, 0.1, param, 'l1', False) for param in params_reg_1]
print(params_reg_1)
print(metrics_reg_1)


sc.stop()