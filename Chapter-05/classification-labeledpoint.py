# coding=utf-8

from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.feature import StandardScaler


conf = SparkConf().setAppName("Classification-LabeledPoint").setMaster("local[2]").set("spark.executor.memory", "5g")
sc = SparkContext(conf= conf)

#get data source
raw_data = sc.textFile("/user-program/python/MachineLearningSpark/Data/train-noheader.tsv")
records = raw_data.map(lambda line : line.split("\t"))
records_first_data = records.first()
print("the first data of records :")
print(records_first_data)
print("the number of records fields :")
print(len(records_first_data))

#get data feature
def labeled_point(r):
    trimmed = map(lambda l : l.replace('\"', ' '), r)
    label = int(trimmed[len(trimmed)-1])
    features = trimmed[4 : len(trimmed)-1]
    features = map(lambda f : f.replace('?', '0'), features)
    for i in range(0, len(features)):
        features[i] = float(features[i])
    return LabeledPoint(label, Vectors.dense(features))

data = records.map(lambda r : labeled_point(r))
num_data = data.count()
print("the number of data :")
print(num_data)

def labeled_point_nb(r):
    trimmed = map(lambda l : l.replace('\"', ' '), r)
    label = int(trimmed[len(trimmed)-1])
    features = trimmed[4 : len(trimmed)-1]
    features = map(lambda f: f.replace('?', '0'), features)
    for i in range(0, len(features)):
        features[i] = float(features[i])
        if features[i] < 0.0:
            features[i] = 0.0
    return LabeledPoint(label, Vectors.dense(features))

nb_data = records.map(lambda r : labeled_point_nb(r))
print("the first data of nb data and the count of nb data:")
print(nb_data.first())

#start train model
num_iterations = 10
max_tree_depth = 5

lr_model = LogisticRegressionWithLBFGS().train(data, num_iterations)
print("logistic regression model :")
print(lr_model)

svm_model = SVMWithSGD().train(data, num_iterations)
print("svm model :")
print(svm_model)

nb_model = NaiveBayes().train(nb_data)
print("naive bayes model :")
print(nb_model)

dt_model = DecisionTree().trainClassifier(data, 2, {})
print("decision tree model :")
print(dt_model)

#start predict
data_point = data.first()
lr_prediction = lr_model.predict(data_point.features)
print("logistic model prediction :" + str(lr_prediction))
print("the true label :" + str(data_point.label))

#analyze data
vectors = data.map(lambda lp : lp.features)
matrix = RowMatrix(vectors)
matrix_summary = matrix.computeColumnSummaryStatistics()
print("the col mean of matrix :")
print(matrix_summary.mean())
print("the col min of matrix :")
print(matrix_summary.min())
print("the col max of matrix :")
print(matrix_summary.max())
print("the col variance of matrix :")
print(matrix_summary.variance())
print("the col num non zero of matrix :")
print(matrix_summary.numNonzeros())

#transform data from data to standard scalar
scaler = StandardScaler(withMean = True, withStd = True).fit(vectors)
labels = data.map(lambda lp : lp.label)
features_transformed = scaler.transform(vectors)

scaled_data = (labels.zip(features_transformed).map(lambda p : LabeledPoint(p[0], p[1])))
print("transformation before :")
print(data.first().features)
print("transformation after :")
print(scaled_data.first().features)

#train logistic regression use scaled data
lr_model_scaled = LogisticRegressionWithLBFGS().train(scaled_data, num_iterations)
print("logistic regression model use scaled data :")
print(lr_model_scaled)
# def total_correct_scaled(sd):
#     if lr_model_scaled.predict(sd.features) == sd.label:
#         return 1
#     else:
#         return 0
# lr_total_correct_scaled = scaled_data.map(lambda sd : total_correct_scaled(sd)).sum()
# print(lr_total_correct_scaled)
# lr_accuracy_scaled = float(lr_total_correct_scaled)/float(num_data)
# print("logistic regression accuracy scaled :")
# print(lr_accuracy_scaled) #the memory is enough


sc.stop()