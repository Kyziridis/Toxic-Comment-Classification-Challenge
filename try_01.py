#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 20:25:58 2018

@author: dead
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, auc
from sklearn.model_selection import train_test_split
import sys
import pickle


def load_data(link):
    # Read csv using pandas 
    train = pd.read_csv(link + "train.csv")
    test = pd.read_csv(link + "test.csv")
    cols_target = ['obscene','insult','toxic','severe_toxic','identity_hate','threat']
    return train, test, cols_target


def preprocess():    
    labels = train.iloc[:,2:]
    text_train = train.iloc[:,1]
    text_valid = valid.iloc[:,1]
    text_test = test.iloc[:,1]
    zero = np.where(np.sum(labels,axis=1)==0)
    # Find the unlabelled percentage
    unlabelled_in_all = train[(train['toxic']!=1) & (train['severe_toxic']!=1) &\
                                 (train['obscene']!=1) & (train['threat']!=1) &\
                                 (train['insult']!=1) & (train['identity_hate']!=1)]
    print("--------------------------------------")
    print("\tData summarize")
    print("--------------------------------------")
    print("Train_set data shape: ", train.shape)
    print("Test__set data shape: ", test.shape)
    print('\nPercentage of unlabelled: ', len(unlabelled_in_all)/len(train)*100)
    print(train[cols_target].sum())
    return text_train, text_valid, text_test



def vectorizing():
    # Tf-IDF vectorizer
    max_feat = 5000
    ngram = (1,2)
    vectorizer = TfidfVectorizer(stop_words='english',\
                                 token_pattern = "\w*[a-z]\w*",\
                                 #analyzer="word",\
                                 # min_df=3,
                                 #max_df=10, 
                                 ngram_range=ngram,\
                                 max_features=max_feat)
    
    print("--------------------------------")
    print("\nCompute TF-IDF...maxFeatures: %s and ngram: %s"%(max_feat,ngram[1]))
    print("Please wait...\n")
    now = time.time()
    
    train_data_features = vectorizer.fit_transform(text_train)
    print("Train_features shape: ", train_data_features.shape)
    
    valid_data_features = vectorizer.transform(text_valid)
    print("Validation_features shape: ", valid_data_features.shape)
    
    test_data_features = vectorizer.transform(text_test)
    print("Test_features shape:  ", test_data_features.shape)
    
    vocab = vectorizer.get_feature_names()
    print("\nTF-IDF completed in %s secs"%(time.time()-now))
    
    #print("\nVOCABULARY for %s maximum features and %s ngrams:"%(max_feat, ngram))
    #print(vocab)

    return train_data_features, valid_data_features, test_data_features
    
    
    
def logistic_classifier():
    # Prepare the LogisticRegression classifier
    logreg = LogisticRegression(C=12.5)
    # Make an empty dataframe to store probabilities results
    logistic_results = pd.DataFrame(columns=cols_target)    
    logistic_valid = pd.DataFrame(columns=cols_target)
    for label in cols_target:
        print('Processing {}'.format(label))
        y = train[label]
        val = valid[label]
        # train the model using train_data_features & y
        logreg.fit(train_data_features, y)
        # compute the training accuracy
        y_pred_X = logreg.predict(train_data_features)
        print('Training accuracy is {}'.format(accuracy_score(y, y_pred_X)))
        # Compute validation accuracy
        val_pred = logreg.predict(valid_data_features)
        val_prob = logreg.predict_proba(valid_data_features)[:,1]
        print('Validation accuracy is {}'.format(accuracy_score(val, val_pred)))        
        # compute the predicted probabilities for X_test_dtm
        test_y_prob = logreg.predict_proba(test_data_features)[:,1]
        logistic_results[label] = test_y_prob
        logistic_valid[label] = val_prob
    return logistic_results, logistic_valid




def SVM():
    
    y = train[cols_target]
    val = valid[cols_target]
    svm_results = pd.DataFrame(columns = cols_target)
    svm_valid = pd.DataFrame(columns = cols_target)
    
    print("Start fitting SVM model..... >_ Support GNU/LINUX >_")
    classif = OneVsRestClassifier(SVC(kernel='linear'))
    classif.fit(train_data_features, y) 
    pickle.dump(classif , "svm_model.pickle" , 'wb')
    
    y_pred_x = classif.predict(train_data_features)
    print('Training accuracy is {}'.format(accuracy_score(y, y_pred_x)))
    
    val_pred = classif.predict(valid_data_features)
    val_prob = classif.predict_proba(valid_data_features)[:,1]
    print('Validation accuracy is {}'.format(accuracy_score(val, val_pred)))
    
    test_y_prob = classif.predict_proba(test_data_features)[:,1]
    svm_results[cols_target] = test_y_prob
    svm_valid[cols_target] = val_prob
    return svm_results, svm_valid
    
    


def ROC():
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for label in cols_target:
        print("-----------------------------")
        print("AUC score for {}: ".format(label) + str(roc_auc_score(valid[label], logistic_valid[label])))
        #print("Score: ",roc_auc_score(valid[label], logistic_valid[label]))       
        fpr[label], tpr[label], _ = roc_curve(valid[label], logistic_valid[label])
        roc_auc[label] = auc(fpr[label], tpr[label])        
    return fpr, tpr, roc_auc        



def ROC_svm():
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for label in cols_target:
        print("-----------------------------")
        print("AUC score for {}: ".format(label) + str(roc_auc_score(valid[label], svm_valid[label])))
        #print("Score: ",roc_auc_score(valid[label], logistic_valid[label]))       
        fpr[label], tpr[label], _ = roc_curve(valid[label], svm_valid[label])
        roc_auc[label] = auc(fpr[label], tpr[label])        
    return fpr, tpr, roc_auc    

        

if __name__ == '__main__':    
     
    path = "/home/dead/Documents/IRTA/project/"   # path for dataset 
    train, test, cols_target = load_data(path)
    # Split the trainSet into train and validation set
    train, valid = train_test_split(train, test_size=0.3, random_state=666)
    # Run preprocess function
    text_train, text_valid ,text_test = preprocess()
    train_data_features, valid_data_features, test_data_features = vectorizing()
    print("Start fitting LogisticRegression classifier")
    print("-------------------------------------------")
    logistic_results, logistic_valid = logistic_classifier()
    logistic_results.to_csv('logistc_results.csv', index=False)
   
    fpr, tpr, roc_auc = ROC()
      
        # PLOTTING
    plt.figure(figsize=(13,8))
    lw = 2
    color = ['blue', 'red', 'darkorange', 'yellow', 'green', 'magenta']
    i = 0
    for l in cols_target:
        plt.plot(fpr[l], tpr[l], color=color[i],
                 lw=lw, label='%s ROC (area = %0.2f)' % (l, roc_auc[l]))
        i = i+1
    
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

    
        
    #svm_results, svm_valid = SVM()
    #svm_results.to_csv("svm_results.csv" , index=False)
    
    #fpr, tpr, roc_auc = ROC_svm()
    
        # PLOTTING
# =============================================================================
#     plt.figure(figsize=(13,8))
#     lw = 2
#     color = ['blue', 'red', 'darkorange', 'yellow', 'green', 'magenta']
#     i = 0
#     for l in cols_target:
#         plt.plot(fpr[l], tpr[l], color=color[i],
#                  lw=lw, label='%s ROC (area = %0.2f)' % (l, roc_auc[l]))
#         i = i+1
#     
#     plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver operating characteristic (ROC)')
#     plt.legend(loc="lower right")
#     plt.show()
# 
# =============================================================================
    
    




# =============================================================================
#     # Barplots    
#     objects = cols_target + ['no_class']
#     y_pos = np.arange(len(objects))
#     plt.figure(figsize=(8,5))
#     plt.bar(y_pos , x , width = 0.8 )
#     plt.xticks(y_pos,objects)
#     plt.ylabel("Frequency")
#     plt.title("Class - Frequencies")
#     plt.show()
# 
# 
#     # Hist
#     train['char_len'] = train['comment_text'].apply(lambda x: len(str(x)))
#     sns.set()
#     train['char_len'].hist()
#     plt.title("Character length")
#     plt.show()
# 
#     
#     # Corr
#     data = train[cols_target]
#     colormap = plt.cm.plasma
#     plt.figure(figsize=(7,7))
#     plt.title('Correlation of features & targets',y=1.05,size=14)
#     sns.heatmap(data.astype(float).corr(),linewidths=0.1,vmax=1.0,square=True,cmap=colormap,
#                linecolor='white',annot=True)
# =============================================================================








































