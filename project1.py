import pandas as pd
import numpy as np
import itertools
from collections import defaultdict
from math import log10
from random import randint
from nltk.corpus import stopwords
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
import string as s
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn import metrics
from matplotlib import pyplot as plt;
from sklearn.metrics import confusion_matrix

def load_data(fname):

  return pd.read_csv(fname)

def generate_challenge_labels(y, uniqname):
  pd.Series(np.array(y)).to_csv(uniqname+'.csv', index=False)
  return

def extract_dictionary(df):
  word_dict = defaultdict(int)
  for index, row in df.iterrows():
    tweet = row["content"]
    for punc in s.punctuation:
      tweet = tweet.replace(punc, ' ')
    words = tweet.lower().split()

    for w in words:
      if w not in word_dict:
        word_dict[w] = len(word_dict)
  return word_dict

def generate_feature_matrix(df, word_dict):
  feature_matrix = np.zeros([len(df), len(word_dict)])

  for index, row in df.iterrows():
    tweet = row["content"]
    for punc in s.punctuation:
      tweet = tweet.replace(punc, ' ')

    words = tweet.lower().split()
    for w in words:
      feature_matrix[index][word_dict[w]] = 1

  return feature_matrix

def cv_performance(clf, X, y, k=5, metric="accuracy"):
  skf = StratifiedKFold(n_splits=k)
  performance_score = []

  for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(X_train, y_train)

    if metric == "auroc":
      y_pred = clf.decision_function(X_test)
    else:
      y_pred = clf.predict(X_test)

    temp_score = performance(y_test, y_pred, metric)
    if not np.isnan(temp_score):
      performance_score.append(temp_score)

  #avg_performance = performance_score / np.float64(k)
  return np.average(performance_score)

def select_param_linear(X, y, k=5, metric="accuracy", C_range = [], penalty='l2'):
  currentMax = -1 * np.float64('inf')
  best_C = C_range[0]
  for c in C_range:
    if penalty == 'l2':
      performance_Score = cv_performance(SVC(kernel='linear',
        C=c, class_weight="balanced"), X, y, k, metric)
    elif penalty == 'l1':
      performance_Score = cv_performance(LinearSVC(penalty='l1',
        dual=False, C=c, class_weight="balanced"), X, y, k, metric)
    if performance_Score > currentMax:
      currentMax = performance_Score
      best_C = c

  #output the result
  print("Performace Measures: " + metric)
  print("Max average 5-fold CV performance:", currentMax)
  print("Parameter C: " + str(best_C))

  return best_C

def select_param_quadratic(X, y, k=5, metric="accuracy", param_range=[]):
  currentMax = -np.float64('inf')
  best_C = 0
  best_R = 0
  for cr_pair in param_range:
    performance_Score = cv_performance(SVC(kernel = 'poly', degree = 2,
      C = cr_pair[0], coef0 = cr_pair[1], class_weight="balanced"), X, y, k, metric)
    if performance_Score > currentMax:
      currentMax = performance_Score
      best_C = cr_pair[0]
      best_R = cr_pair[1]

  #output the result
  print("Performace Measures: " + metric)
  print("Max average 5-fold CV performance:", currentMax)
  print("Parameter C: " + str(best_C))
  print("Parameter R: " + str(best_R))
  return best_C, best_R


def performance(y_true, y_pred, metric="accuracy"):
  if metric == 'accuracy':
    return np.float64(metrics.accuracy_score(y_true, y_pred))
  elif metric == 'f1-score':
    return np.float64(metrics.f1_score(y_true, y_pred))
  elif metric == 'auroc':
    return np.float64(metrics.roc_auc_score(y_true, y_pred))
  elif metric == 'precision':
    return np.float64(metrics.precision_score(y_true, y_pred))

  # used for two other measures
  confusionMatrix = metrics.confusion_matrix(y_true, y_pred, [1,-1])
  TP = confusionMatrix[0][0]
  FN = confusionMatrix[0][1]
  FP = confusionMatrix[1][0]
  TN = confusionMatrix[1][1]

  if metric == 'sensitivity':
    return np.float64(TP) / np.float64(TP + FN)
  elif metric == 'specificity':
    return np.float64(TN) / np.float64(FP + TN)


def performance_CI(clf, X, y, metric="accuracy"):
  if metric == "auroc":
    performance_result = performance(y, clf.decision_function(X), metric)
  else:
    performance_result = performance(y, clf.predict(X), metric)

  bootstrap_Samples = []
  for i in range(1000):
      sample = np.random.randint(len(y), size=len(y))
      if metric == "auroc":
        temp_performance = performance(y[sample], clf.decision_function(X[sample]), metric)
      else:
        temp_performance = performance(y[sample], clf.predict(X[sample]), metric)

      if not np.isnan(temp_performance):
          bootstrap_Samples.append(temp_performance)

  lower = np.percentile(bootstrap_Samples, 2.5)
  upper = np.percentile(bootstrap_Samples, 97.5)
  return performance_result, lower, upper


def extract_dictionary_part5(dict, df):
  word_dict = dict
  #ignore the words which do not contain important significance
  stopword_list = set(stopwords.words("english"))

  for idx, row in df.iterrows():
    tweet = row["content"]
    #do not count name
    if '@' in tweet:
      tweet = " ".join(filter(lambda x:x[0]!='@', tweet.split()))
    for punc in s.punctuation:
      tweet = tweet.replace(punc, ' ')
    words = tweet.lower().split()

    for w in words:
      if w not in dict and w not in stopword_list:
        word_dict[w] = len(word_dict)

  return word_dict

def generate_feature_matrix_part5(df, word_dict):
  feature_matrix = np.zeros([len(df), len(word_dict)])
  for idx, row in df.iterrows():
    tweet = row["content"]
    for punc in s.punctuation:
      tweet = tweet.replace(punc, ' ')

    words = tweet.lower().split()
    for w in words:
      if w in word_dict:
        #Using the number of times a word occurs
        feature_matrix[idx][word_dict[w]] += 1

  return feature_matrix


def main():
  '''
  print("part2: ")
  df = load_data("dataset.csv")
  words_dict = extract_dictionary(df)
  feature_matrix = generate_feature_matrix(df, words_dict)
  print("The value of d: " + str(len(words_dict)))

  #calculate the average nonzero
  count = 0
  for row in range(feature_matrix.shape[0]):
    for col in range(feature_matrix.shape[1]):
      if feature_matrix[row][col] == 1:
        count += 1

  print("The average number of non-zero features per tweet: " + str(count / len(df)))


  print("part3: ")
  k = 5
  #cutoff the test set, only keep the training set
  train_SetNum = 400
  #set the X and y
  y = df['label'][:400]
  X = feature_matrix[:train_SetNum]

  metrics = ["accuracy", 'f1-score', 'auroc', 'precision', 'sensitivity', 'specificity']
  C_range = [10**c for c in range(-3,4)]
  print("C range is: ")
  print(C_range)

  #3.1(c)
  print("***Result for 3.1(c)")
  for m in metrics:
    c = select_param_linear(X, y, 5, m, C_range)


  #3.1(d)
  #calculate L0-norm
  L0_form_list = []
  for c in C_range:
    clf = SVC(kernel = 'linear', C = c, class_weight = "balanced")
    clf.fit(X, y)
    theta = clf._get_coef()

    L0_form = 0
    for i in range(theta.size):
      if theta[0][i] != 0:
        L0_form += 1

    #print("C: " + str(c))
    #print("L0-form: " + str(L0_form))
    L0_form_list.append(L0_form)

  plt.figure()
  plt.plot(C_range, L0_form_list, '-or', label='L0-norm')
  plt.xlabel('C')
  plt.ylabel('L0')
  plt.xscale('log')
  plt.show()

  #3.2
  R_range = C_range
  #grid search
  #generate the param_range
  param_range = []
  for c in C_range:
    for r in R_range:
      param_range.append([c, r])

  print("***Result for 3.2 grid search")
  c, r = select_param_quadratic(X, y, 5, 'auroc', param_range)


  #random search
  print("***Result for 3.2 random search")
  pair_Num = 25

  param_range = []
  for i in range(pair_Num):
      c = np.random.uniform(-3,3)
      r = np.random.uniform(-3,3)
      param_range.append((pow(10, c), pow(10, r)))

  c, r = select_param_quadratic(X, y, 5, 'auroc', param_range)


  print("***Result of 3.4(a) ")
  c = select_param_linear(X, y, 5, "auroc", C_range, "l1")

  print("***Result of 3.4(b) ")
  #calculate L0-norm
  L0_form_list = []
  for c in C_range:
      clf = LinearSVC(penalty='l1', C=c,dual=False, class_weight='balanced')
      clf.fit(X, y)
      theta = clf.coef_

      L0_form = 0
      for i in range(theta.size):
        if theta[0][i] != 0:
          L0_form += 1

      #print("C: " + str(c))
      #print("L0-form: " + str(L0_form))
      L0_form_list.append(L0_form)

  plt.figure()
  plt.plot(C_range, L0_form_list, '-or', label='L0-norm')
  plt.xlabel('C')
  plt.ylabel('L0')
  plt.xscale('log')
  plt.show()

  #3.5 choose parameter
  #get test set
  y_test = df['label'][400:].reset_index(drop=True)
  X_test = feature_matrix[400:]

  linear_l2_C = 0.1
  clf1 = SVC(kernel='linear', C=linear_l2_C, class_weight='balanced')
  clf1.fit(X, y)

  linear_l1_C = 100
  clf2 = LinearSVC(penalty='l1', C=linear_l1_C, dual=False, class_weight='balanced')
  clf2.fit(X, y)

  quadratic_l2_CR = [1000, 0.1]
  clf3 = SVC(kernel='poly', degree=2, C=quadratic_l2_CR[0], coef0=quadratic_l2_CR[1], class_weight='balanced')
  clf3.fit(X, y)

  print("***Result of part 4")
  #linear_l2
  performance, lower, upper = performance_CI(clf1, X_test, y_test, metric="auroc")
  print("Liner_hinge_L2, Performance = %f, CI is [%f, %f]" % (performance,lower,upper))

  #linear_l1
  performance, lower, upper=performance_CI(clf2, X_test, y_test, metric="auroc")
  print("Liner_square_L1, Performance = %f, CI is [%f, %f]" % (performance,lower,upper))

  #quadratic_l2
  performance, lower, upper=performance_CI(clf3, X_test, y_test, metric="auroc")
  print("Quad_hinge_L2, Performance = %f, CI is [%f, %f]" % (performance,lower,upper))
  '''
  #part 5
  trainingData = load_data("challenge.csv")
  testData = load_data("held_out.csv")
  dict = defaultdict(int)
  dict = extract_dictionary_part5(dict, trainingData)
  dict = extract_dictionary_part5(dict, testData)

  X = generate_feature_matrix_part5(trainingData, dict)
  X_test = generate_feature_matrix_part5(testData, dict)

  #set the true labels 0 is sadness
  y = np.zeros(X.shape[0])
  for idx, row in trainingData.iterrows():
      if row["sentiment"] == "hate":
          y[idx] = -1
      if row["sentiment"] == "love":
          y[idx] = 1

  y_pred = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X)
  confusionMatrix = metrics.confusion_matrix(y, y_pred, [-1, 0, 1])
  print("***Result of part 5, confusion matrix")
  print(confusionMatrix)
  y_pred = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X_test)
  generate_challenge_labels(y_pred, "lchenhao")

  return

if __name__ == '__main__':
  main()



