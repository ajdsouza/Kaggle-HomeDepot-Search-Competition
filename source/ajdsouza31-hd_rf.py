#
# ajdsouza31 - dl
#
# Submitted as final project work project ISYE-7406Q
#
# Kaggle Home Depot competition
#
# This file reads the data the featurized learnign data and performs learning
#
#-----------------------------------------------------------------------------------------------------
# THis part of the code is to fit the model by reading the saved feature vector and 
#  fitting a model to it using gridsearchcv
#-----------------------------------------------------------------------------------------------------
import time
time0 = time.time()

import pprint as pp
import numpy as np
import pandas as pd
import re


from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import pipeline, grid_search
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, make_scorer
import xgboost as xgb
import sklearn.linear_model as lm
from sklearn.base import clone
from sklearn.base import BaseEstimator

from nltk.stem.porter import *
stemmer = PorterStemmer()
#from nltk.stem.snowball import SnowballStemmer #0.003 improvement but takes twice as long as PorterStemmer
#stemmer = SnowballStemmer('english')

# default one
crandomseed = 20160417
randomseed = int(time0)

import random
random.seed(randomseed)

from sklearn.cross_validation import train_test_split
import os
from sklearn.externals import joblib

import StringIO
import shutil
import json
from scipy.sparse.linalg import svds
import operator
import os
import glob
import matplotlib.pyplot as plt
from sklearn import cross_validation as bst
from cycler import cycler
import scipy.stats as stats
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pickle

#---------------------------------------------------------------------
# Support functions
#---------------------------------------------------------------------
# save model results
all_results  = {}

# read the last all_results file from dir and save that as all_results
all_result_files = [file for file in glob.glob(os.path.join(*[os.getcwd(), 'data_pkl', '*all_results.json']))]
if all_result_files:
    all_result_files.sort(key=os.path.getmtime)
    with open(all_result_files[-1]) as all_results_latest:
        all_results = json.load(all_results_latest)

# get the closest relevance score from the relvals list
def get_closest_relval(n):
    global relvals
    return min(relvals, key=lambda x:abs(x-n))


# fix outof range predictions
def fix_xgb_range(mstring,y_p):
    min_y_pred = min(y_p)
    max_y_pred = max(y_p)
    print(mstring,min_y_pred, max_y_pred)
    for i in range(len(y_p)):
        if y_p[i]<1.0:
            y_p[i] = 1.0
        if y_p[i]>3.0:
            y_p[i] = 3.0
    return y_p


#save model results
def save_model_results(X_train,
                       y_train,
                       X_model_test, 
                       y_model_test,
                       train_df,
                       train_df_relevance,                       
                       X_test,
                       mstring,
                       clf,
                       model=None):
    
    global all_results
    global id_test
    global time0
        
    # timestamp to get files of a model together
    ts = str(int(time.time()))
    mstring = mstring + '_' + ts
    print("In save_model_Results for "+mstring)
    
    # for this trained model create a directory to save model information
    if not os.path.exists(ts):
        os.makedirs(ts)
    
    if model is not None:
        model.fit(X_train, y_train)
        print(mstring+"================== Best Param Model Report ========================")
        print(mstring+"Best Training Estimator")
        pp.pprint(model.estimator)
        print(mstring+"Best Params")
        pp.pprint(model.best_params_)
        print(mstring+"Best CV Score training: %r" % model.best_score_)
        print(mstring+"Best CV Score corrected training %r" % (model.best_score_ + 0.47003199274))
        y_model_pred = model.predict(X_model_test)
    else:
        clf.fit(X_train, y_train)
        y_model_pred = clf.predict(X_model_test)
    
    if 'xgb' in mstring:
        y_model_pred = fix_xgb_range(mstring,y_model_pred)
    
    print(">>>>>>>>>%s time to train on training data with cv to get best params %d" % (mstring, round(((time.time() - time0)/60),2)) )
    
    #----------------------------------------------------------------------------
    # Use the best parameter model to predict on the held back test data
    #
    # Use these results to evaluate different models before submission
    #
    # ALso use this for project resutls
    #----------------------------------------------------------------------------
    y_model_pred_closest =  [ get_closest_relval(n) for n in y_model_pred ]
    y_model_rmse = fmean_squared_error(y_model_test,y_model_pred)
    y_model_rmse_closest = fmean_squared_error(y_model_test,y_model_pred_closest)
    
    print(mstring+"RMSE for model test data %r" % y_model_rmse)
    print(mstring+"RMSE for model test data closest %r" % y_model_rmse_closest)
    
    print(">>>>>>>>>%s Time to predict the best cv model on the model test data %d" % (mstring, round(((time.time() - time0)/60),2)))
        
    #----------------------------------------------------------------------------
    # Fit the best params model on 60% of the training data and 
    # use that to evaluate performance on the remaining 40% train data
    #----------------------------------------------------------------------------
    if model is not None:
        best_params = model.best_params_
        clf.set_params(**best_params)
    
    X_ptrain, X_ptest, y_ptrain, y_ptest = train_test_split(train_df, train_df_relevance, 
                                                            test_size=0.6, random_state=randomseed)
    clf.fit(X_ptrain,  y_ptrain)
    y_ptrain_pred = clf.predict(X_ptest)
    if 'xgb' in mstring:
        y_ptrain_pred = fix_xgb_range(mstring,y_ptrain_pred)
    
    y_ptrain_pred_closest = [ get_closest_relval(n) for n in y_ptrain_pred ]
    y_train_refit_rmse = fmean_squared_error(y_ptest, y_ptrain_pred)
    y_train_refit_rmse_closest = fmean_squared_error(y_ptest, y_ptrain_pred_closest)
    
    print("****************** Refit on Train Data Double check of Fit *************************")
    print("%s RMSE for refit on whole trained data %r" % (mstring, y_train_refit_rmse))
    print("%s RMSE for refit on whole trained data closest %r" % (mstring, y_train_refit_rmse_closest))
    
    
    print(">>>>>>>>>%s Time to train the model with best params on whole training data %d" % (mstring, round(((time.time() - time0)/60),2)))
    
    # save the model information
    output = StringIO.StringIO()
    
    output.write(mstring)
    output.write(os.linesep)
    output.write(str(clf))
    output.write(os.linesep)
    
    if model is not None:
        output.write(model.best_params_)
        output.write(os.linesep)
        output.write(model.best_score_)
        output.write(os.linesep)
        output.write((model.best_score_ + 0.47003199274))
        output.write(os.linesep)
        output.write(y_model_rmse)
        output.write(os.linesep)
        output.write(y_model_rmse_closest)
        output.write(os.linesep)
    
    output.write(y_train_refit_rmse)
    output.write(os.linesep)
    output.write(y_train_refit_rmse_closest)
    output.write(os.linesep)
    
    with open (ts+'/model.txt', 'w') as fd:
      output.seek (0)
      shutil.copyfileobj (output, fd)
    
    output.close()
    
    #----------------------------------------------------------------------------
    # create the submission data
    #
    # Predict on test and save the results for submission
    #----------------------------------------------------------------------------
    #----------------------------------------------------------------------------
    # Fit the best params model on the whole training data and 
    # use that to evaluate submission
    #----------------------------------------------------------------------------
    clf.fit(train_df,  train_df_relevance)
    y_pred = clf.predict(X_test)
    if 'xgb' in mstring:
        y_pred = fix_xgb_range(mstring,y_pred)
    
    y_pred_closest = [ get_closest_relval(n) for n in y_pred ]
    
    pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv(ts+'/submission_'+ ts +'.csv',index=False)
    pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv(ts+'/submission_cl_'+ ts +'.csv',index=False)
    
    print(">>>>>>>>>Time tp make a prediction for submission test the model %d" % round(((time.time() - time0)/60),2))
    
    #----------------------------------------------------------------------
    # save the model
    #---------------------------------------------------------------------
    joblib.dump(clf, ts+'/filename'+ ts +'.pkl') 
    
    #----------------------------------------------------------------------
    # Save the results for printing
    #---------------------------------------------------------------------
    if 'rmse_train_refit' not in all_results:
        all_results['rmse_train_refit'] = {}
    if 'rmse_train_refit_closest' not in all_results:
        all_results['rmse_train_refit_closest'] = {}
    if 'rmse_test_data' not in all_results:
        all_results['rmse_test_data'] = {}
    if 'rmse_test_data_closest' not in all_results:
        all_results['rmse_test_data_closest'] = {}
    if 'rmse_best_cv_score' not in all_results:
        all_results['rmse_best_cv_score'] = {}
    if 'cv_best_params' not in all_results:
        all_results['cv_best_params'] = {}
    
    all_results['rmse_train_refit'][mstring] = y_train_refit_rmse
    all_results['rmse_train_refit_closest'][mstring] = y_train_refit_rmse_closest
    all_results['rmse_test_data'][mstring] = y_model_rmse
    all_results['rmse_test_data_closest'][mstring] = y_model_rmse_closest
    all_results[mstring] = {}
    all_results[mstring]['randomseed'] = randomseed
    if model is not None:
        all_results['rmse_best_cv_score'][mstring] = model.best_score_
        all_results['cv_best_params'][mstring] = model.best_params_



#----------------------------------------------------------------------------------------
# read the saved data and split it training set and model test set
#  The training data is 75k, the transformed data has over25K features
#   to run a 10 fold cross validation to choose paramater combinations
#    in the range of 8000 permutations takes 80000 iterations over 75K rows
#
# TO make this work on a single cpu I decided to 
# 1. perform cv only on 10% of the training set - 75K rows
# 2. Use the remaining 90% of the training data as held back test data
# 3. Make a prediction for the best param model from cv on the held back test data
# 4. Use the RMSE score on the test data to compare models( can bootstrap here to reduce variance)
# 5. The model with the best RMSE score, train it only the whole test data with the best
#    paramets chosen earlier in cv
# 6. Use the best model trained on the whole data set as the one to make a prediction 
#    on the competion test data 
# 7. Submit this data to the competition
#----------------------------------------------------------------------------------------
train_df = pd.read_csv('train_df.csv', encoding="ISO-8859-1", index_col=0)

# remove rows which have relevance scores count > 2, 59/74K outliers <.001%
relv_frequency = dict(train_df.groupby('relevance')['relevance'].count())

relvals = [k for k, v in relv_frequency.items() if v >= 20]

# remove values where relevance count < 20 , as outlier data
train_df = train_df[train_df['relevance'].isin(relvals)]

#----------------------------------------------------------------------------
# create the submission data
#
# read the saved test data and
# Predict on test and save the results for submission
#----------------------------------------------------------------------------
test_df = pd.read_csv('test_df.csv', encoding="ISO-8859-1", index_col=0)

#--------------------------------------------------------------------
# train the model
#--------------------------------------------------------------------

# define the support classes and function fro grid search cv with pipelining

# SCORING function
# RMSE using sklearn.metrics.mean_squared_error
def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = np.abs(mean_squared_error(ground_truth, predictions)**0.5)
    return fmean_squared_error_


# create callable for scorer to pass to grid search cv
RMSE  = make_scorer(fmean_squared_error, greater_is_better=False)

#
# first step of the pipeline
#
# FIT returns self and 
# TRANSFORM to remove the pure text columns which are not feature vectors and return df
#
d_col_drops=['id','relevance','search_term','product_title',
             'product_description','product_info','attr',
             'brand','search_term_segs_in_title']

# get feature names for the columns
feature_names = np.array([col for col in train_df.columns.values if col not in d_col_drops])

class cust_regression_vals(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, hd_searches):
        hd_searches = hd_searches.drop(d_col_drops,axis=1).values
        return hd_searches
    def get_feature_names(self):
        global feature_names
        return feature_names


#
# first step within each nested pipeline for getting the tfidf svd matrix for text fields
#
# FIT returns self
# and transform returns the data for the requested text column
#
class cust_txt_col(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, data_dict):
        return data_dict[self.key].apply(str)


#
# first step within each nested pipeline for getting the tfidf svd matrix for text fields
#
# FIT returns self
# and transform returns the data for the requested text column
#

# get the rank of a matrix x

eps = 1e-12
def rank(A):
    u, s, vh = svds(A)
    return len([x for x in s if abs(x) > eps])


# perform truncated svd and choose rank and pass it to n components
# class cust_tsvd(BaseEstimator, TransformerMixin):
class cust_tsvd(BaseEstimator, TransformerMixin):
    def __init__(self,key=None):
        self.key = key
        self.tsvd = TruncatedSVD(random_state = randomseed)
        
    def fit(self, x, y=None):
        rk = rank(x)
        print("Rank of matrix %r x %r is %r" % (x.shape[0], x.shape[1], rk))
        params_svd = {"n_components":rk}
        self.tsvd.set_params(**params_svd)
        self.tsvd.fit(x)
        return self
    
    def transform(self, x):
        return self.tsvd.transform(x)
    
    def set_params(self,p):
        return self.tsvd.set_params(p)
    
    def get_params(self,deep=True):
        return self.tsvd.get_params()




tsvd = cust_tsvd()
# tsvd gives a runtime error so just stick to ysing TruncatedSVD directly as this is PCA
#  not a rank reduction
ncomp_svd = 10
tsvd = TruncatedSVD(n_components=ncomp_svd,random_state = randomseed)

# add feature names for the ncomp_svd svd components to be added for each of these tfidf columns
for col in ('search_term_segs_in_title','product_title','product_description','brand'):
    for i in range(ncomp_svd+1):
        feature_names = np.append(feature_names,col+"_tfidf_svd_comp_"+str(i))

#comment out the lines below use all_data_df.csv for further grid search testing
#if adding features consider any drops on the 'cust_regression_vals' class
#*** would be nice to have a file reuse option or script chaining option on Kaggle Scripts ***

id_test = test_df['id']
X_test = test_df[:]

test_fraction = 0.80

X_train, X_model_test, y_train, y_model_test = train_test_split(train_df, train_df['relevance'].values, 
                                                                test_size=test_fraction, random_state=randomseed)
    
xgbr = None
gbr = None
rfr = None
laso = None

#for i in [1,2,3]:
for i in [6]:
    
    print("IN "+str(i))
    mstring = ''    
    cv = 3
    
    X_train, X_model_test, y_train, y_model_test = train_test_split(train_df, train_df['relevance'].values, 
                                                                    test_size=test_fraction, random_state=randomseed)
    
    if i == 1:
        mstring = 'rfr cv = ' +str(cv)
    elif i == 2:
        mstring = 'gbr cv = ' +str(cv)  
    elif i == 3:
        mstring = 'xgb cv = ' +str(cv)
    elif i == 4:
        mstring = 'lasso cv = ' +str(cv)    
    elif i == 5:
        cv = -1
        mstring = 'rfr cv = ' +str(cv)    
    elif i == 6:
        cv = -1
        mstring = 'gbr cv = ' +str(cv)    
    elif i == 7:
        cv = -1
        mstring = 'xgb cv = ' +str(cv) 
    elif i == 8:
        cv = -1
        mstring = 'lasso cv = ' +str(cv)    
    
    print('Performing '+mstring)
        
    tfidf = TfidfVectorizer(stop_words='english')
   
    # 1 to 3 grams
    ngram_tfidf = [(1,3)]
    transformer_weights = [ [1.0, 0.75, 0.50, 0.15, 1 ] ] 
     
    # to get the list of parameter with tags in the estimator
    #   pp.pprint(sorted(clf.get_params().keys()))
    if 'rfr' in mstring:
        
        rgr = RandomForestRegressor(n_jobs = 1, random_state = randomseed, verbose = 1)
        
        #cv = 1
        if cv == 1:
            param_grid = {
                'union__transformer_weights': transformer_weights[0],
                'rgr__max_features': ["auto"][0], 
                'rgr__max_leaf_nodes': [4,15,30][0],
                'rgr__min_samples_split': [1][0],
                'rgr__oob_score' : [1][0],
                'rgr__n_estimators' : [300,600,800][0],
                'union__txt1__tfidf1__ngram_range': ngram_tfidf[0],
                'union__txt1__tsvd1__n_components': [ncomp_svd][0],
                'union__txt2__tfidf2__ngram_range': ngram_tfidf[0],
                'union__txt2__tsvd2__n_components': [ncomp_svd][0],
                'union__txt3__tfidf3__ngram_range': ngram_tfidf[0],
                'union__txt3__tsvd3__n_components': [ncomp_svd][0],
                'union__txt4__tfidf4__ngram_range': ngram_tfidf[0],
                'union__txt4__tsvd4__n_components': [ncomp_svd][0]
                }
        #cv == -1
        elif cv == -1:
            rgr_params =  {
                'max_features' : ["auto"][0],
                'max_leaf_nodes': [4][0],
                'min_samples_split': [1][0],
                'oob_score': [1][0],
                'n_estimators' : [1000][0]
                }
            rgr.set_params(**rgr_params)            
            xparam_grid = {
                'transformer_weights': transformer_weights[0],
                'txt1__tfidf1__ngram_range': ngram_tfidf[0],
                'txt1__tsvd1__n_components': [ncomp_svd][0],
                'txt2__tfidf2__ngram_range': ngram_tfidf[0],
                'txt2__tsvd2__n_components': [ncomp_svd][0],
                'txt3__tfidf3__ngram_range': ngram_tfidf[0],
                'txt3__tsvd3__n_components': [ncomp_svd][0],
                'txt4__tfidf4__ngram_range': ngram_tfidf[0],
                'txt4__tsvd4__n_components': [ncomp_svd][0]        
                }
            # save it for plotting performance of the GradientBoostRegression
            rfr = rgr
        else:
            
            param_grid = {
                'union__transformer_weights': transformer_weights,
                'rgr__max_features': ["auto"], 
                'rgr__max_leaf_nodes': [4,10,20],
                'rgr__min_samples_split': [1],
                'rgr__oob_score' : [1],
                'rgr__n_estimators' : [100,400,800],
                'union__txt1__tfidf1__ngram_range': ngram_tfidf,
                'union__txt1__tsvd1__n_components': [ncomp_svd],
                'union__txt2__tfidf2__ngram_range': ngram_tfidf,
                'union__txt2__tsvd2__n_components': [ncomp_svd],
                'union__txt3__tfidf3__ngram_range': ngram_tfidf,
                'union__txt3__tsvd3__n_components': [ncomp_svd],
                'union__txt4__tfidf4__ngram_range': ngram_tfidf,
                'union__txt4__tsvd4__n_components': [ncomp_svd]
                }
    
    elif 'gbr' in mstring:
        
        rgr = GradientBoostingRegressor(random_state = randomseed, verbose = 1)
        
        #cv = 1
        if cv == 1:
            param_grid = {
                'union__transformer_weights': transformer_weights[0],
                'rgr__max_features': ["auto"][0], 
                'rgr__max_leaf_nodes': [5][0],
                'rgr__min_samples_split': [1][0],
                'rgr__n_estimators' : [500][0],
                'rgr__learning_rate' : [0.05][0],
                'rgr__loss': ['ls'][0],
                'rgr__subsample': [0.5][0],
                'rgr__warm_start': [False][0],
                'union__txt1__tfidf1__ngram_range': ngram_tfidf[0],
                'union__txt1__tsvd1__n_components': [ncomp_svd][0],
                'union__txt2__tfidf2__ngram_range': ngram_tfidf[0],
                'union__txt2__tsvd2__n_components': [ncomp_svd][0],
                'union__txt3__tfidf3__ngram_range': ngram_tfidf[0],
                'union__txt3__tsvd3__n_components': [ncomp_svd][0],
                'union__txt4__tfidf4__ngram_range': ngram_tfidf[0],
                'union__txt4__tsvd4__n_components': [ncomp_svd][0]
                }
        #cv == -1
        elif cv == -1:
            rgr_params =  {
                'max_features': ["auto"][0],
                'max_leaf_nodes': [5][0],
                'min_samples_split': [1][0],
                'n_estimators' : [1000][0],
                'learning_rate' : [0.05][0],
                'loss': ['ls'][0],
                'subsample': [1.0][0],
                'warm_start': [False][0] }
            rgr.set_params(**rgr_params)            
            xparam_grid = {
                'transformer_weights': transformer_weights[0],
                'txt1__tfidf1__ngram_range': ngram_tfidf[0],
                'txt1__tsvd1__n_components': [ncomp_svd][0],
                'txt2__tfidf2__ngram_range': ngram_tfidf[0],
                'txt2__tsvd2__n_components': [ncomp_svd][0],
                'txt3__tfidf3__ngram_range': ngram_tfidf[0],
                'txt3__tsvd3__n_components': [ncomp_svd][0],
                'txt4__tfidf4__ngram_range': ngram_tfidf[0],
                'txt4__tsvd4__n_components': [ncomp_svd][0]        
                }
            # save it for plotting performance of the GradientBoostRegression
            gbr = rgr
        else:
            param_grid = {
                'union__transformer_weights': transformer_weights,
                'rgr__max_features': ["auto"], 
                'rgr__max_leaf_nodes': [4,5],
                'rgr__min_samples_split': [1],
                'rgr__n_estimators' : [400,500],
                'rgr__learning_rate' : [0.05,0.01],
                'rgr__loss': ['ls'],
                'rgr__subsample': [1.0,0.5],
                'rgr__warm_start': [False],
                'union__txt1__tfidf1__ngram_range': ngram_tfidf,
                'union__txt1__tsvd1__n_components': [ncomp_svd],
                'union__txt2__tfidf2__ngram_range': ngram_tfidf,
                'union__txt2__tsvd2__n_components': [ncomp_svd],
                'union__txt3__tfidf3__ngram_range': ngram_tfidf,
                'union__txt3__tsvd3__n_components': [ncomp_svd],
                'union__txt4__tfidf4__ngram_range': ngram_tfidf,
                'union__txt4__tsvd4__n_components': [ncomp_svd]
                }
    
    elif 'xgb' in mstring:
        
        rgr = xgb.XGBRegressor(silent=False, objective="reg:linear", nthread=1, gamma=0, min_child_weight=1, max_delta_step=0,
                               colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                               base_score=0.5, seed=0, missing=None)
        
        # handle any categorical data for xgb
        train2 = train_df[d_col_drops]
        test2 = test_df[d_col_drops]
        train = train_df.drop(d_col_drops,axis=1)[:]
        test =  test_df.drop(d_col_drops,axis=1)[:]
            
        for (train_name, train_series), (test_name, test_series) in zip(train.iteritems(),test.iteritems()):
            if train_series.dtype == 'O':
                train[train_name], tmp_indexer = pd.factorize(train[train_name])
                test[test_name] = tmp_indexer.get_indexer(test[test_name])
            else:
                tmp_len = len(train[train_series.isnull()])
                if tmp_len>0:
                    train.loc[train_series.isnull(), train_name] = train_series.mean()                    
                tmp_len = len(test[test_series.isnull()])
                if tmp_len>0:
                    test.loc[test_series.isnull(), test_name] = train_series.mean() 
        
        train_df=pd.concat([train,train2], axis=1)[:]                        
        test_df=pd.concat([test,test2], axis=1)[:]
        X_test = test_df[:]
        
        # split again as we have modified train_df for xgb
        X_train, X_model_test, y_train, y_model_test = train_test_split(train_df, train_df['relevance'].values, 
                                                                        test_size=test_fraction, random_state=randomseed)
        
        
        #cv = 1
        if cv == 1:
            param_grid = {
                'union__transformer_weights': transformer_weights[0],
                'rgr__max_depth': [3][0],
                'rgr__n_estimators' : [400][0],
                'rgr__learning_rate' : [0.05][0],
                'rgr__subsample': [1.0][0],
                'rgr__gamma' :[0.0][0],
                'union__txt1__tfidf1__ngram_range': ngram_tfidf[0],
                'union__txt1__tsvd1__n_components': [ncomp_svd][0],
                'union__txt2__tfidf2__ngram_range': ngram_tfidf[0],
                'union__txt2__tsvd2__n_components': [ncomp_svd][0],
                'union__txt3__tfidf3__ngram_range': ngram_tfidf[0],
                'union__txt3__tsvd3__n_components': [ncomp_svd][0],
                'union__txt4__tfidf4__ngram_range': ngram_tfidf[0],
                'union__txt4__tsvd4__n_components': [ncomp_svd][0]
                }
        #cv == -1
        elif cv == -1:
            rgr_params =  {
                'max_depth': [5][0],
                'n_estimators' : [600][0],
                'learning_rate' : [0.01][0],
                'subsample': [1.0][0],
                'gamma' :[0.0][0]}
            rgr.set_params(**rgr_params)
            xparam_grid = {
                'transformer_weights': transformer_weights[0],
                'txt1__tfidf1__ngram_range': ngram_tfidf[0],
                'txt1__tsvd1__n_components': [ncomp_svd][0],
                'txt2__tfidf2__ngram_range': ngram_tfidf[0],
                'txt2__tsvd2__n_components': [ncomp_svd][0],
                'txt3__tfidf3__ngram_range': ngram_tfidf[0],
                'txt3__tsvd3__n_components': [ncomp_svd][0],
                'txt4__tfidf4__ngram_range': ngram_tfidf[0],
                'txt4__tsvd4__n_components': [ncomp_svd][0]     
                }
            # save it for plotting performance of the GradientBoostRegression
            xgbr = rgr
        else:
            cv = 3
            param_grid = {
                'union__transformer_weights': transformer_weights,
                'rgr__max_depth': [5],
                'rgr__n_estimators' : [200,400,600,1000],
                'rgr__learning_rate' : [0.01,0.1,0.2],
                'rgr__subsample': [1.0],
                'rgr__gamma' :[0,0.01,0.2,0.5],
                'union__txt1__tfidf1__ngram_range': ngram_tfidf,
                'union__txt1__tsvd1__n_components': [ncomp_svd],
                'union__txt2__tfidf2__ngram_range': ngram_tfidf,
                'union__txt2__tsvd2__n_components': [ncomp_svd],
                'union__txt3__tfidf3__ngram_range': ngram_tfidf,
                'union__txt3__tsvd3__n_components': [ncomp_svd],
                'union__txt4__tfidf4__ngram_range': ngram_tfidf,
                'union__txt4__tsvd4__n_components': [ncomp_svd]
                }
            
    # LassoCV
    elif 'lasso' in mstring:               
        #cv = 1
        if cv == 1:
            rgr = lm.Lasso(random_state = randomseed)
            param_grid = {
                'union__transformer_weights': transformer_weights[0],
                'rgr__fit_intercept': [True,False][0], 
                'rgr__normalize': [False,True][0],
                'rgr__alpha': [0.001,0.1][0],
                'union__txt1__tfidf1__ngram_range': ngram_tfidf[0],
                'union__txt1__tsvd1__n_components': [ncomp_svd][0],
                'union__txt2__tfidf2__ngram_range': ngram_tfidf[0],
                'union__txt2__tsvd2__n_components': [ncomp_svd][0],
                'union__txt3__tfidf3__ngram_range': ngram_tfidf[0],
                'union__txt3__tsvd3__n_components': [ncomp_svd][0],
                'union__txt4__tfidf4__ngram_range': ngram_tfidf[0],
                'union__txt4__tsvd4__n_components': [ncomp_svd][0]
                }
        #cv == -1
        elif cv == -1:
            rgr = lm.Lasso(random_state = randomseed)
            rgr_params =  {
                'fit_intercept': [True][0],
                'alpha' : [0.001][0],
                'normalize' : [False][0] }
            rgr.set_params(**rgr_params)
            xparam_grid = {
                'transformer_weights': transformer_weights[0],
                'txt1__tfidf1__ngram_range': ngram_tfidf[0],
                'txt1__tsvd1__n_components': [ncomp_svd][0],
                'txt2__tfidf2__ngram_range': ngram_tfidf[0],
                'txt2__tsvd2__n_components': [ncomp_svd][0],
                'txt3__tfidf3__ngram_range': ngram_tfidf[0],
                'txt3__tsvd3__n_components': [ncomp_svd][0],
                'txt4__tfidf4__ngram_range': ngram_tfidf[0],
                'txt4__tsvd4__n_components': [ncomp_svd][0]     
                }
            # save it for plotting performance of the GradientBoostRegression
            laso = rgr
        else:            
            rgr = lm.LassoCV(n_jobs=1, random_state = randomseed, verbose = 1)
            param_grid = {
                'union__transformer_weights': transformer_weights,
                'rgr__fit_intercept': [True,False], 
                'rgr__normalize': [True,False],
                'rgr__n_alphas' : [np.arange(.001,1,.005).shape[0]],
                'rgr__alphas' : [np.arange(.001,1,.005)],
                'union__txt1__tfidf1__ngram_range': ngram_tfidf,
                'union__txt1__tsvd1__n_components': [ncomp_svd],
                'union__txt2__tfidf2__ngram_range': ngram_tfidf,
                'union__txt2__tsvd2__n_components': [ncomp_svd],
                'union__txt3__tfidf3__ngram_range': ngram_tfidf,
                'union__txt3__tsvd3__n_components': [ncomp_svd],
                'union__txt4__tfidf4__ngram_range': ngram_tfidf,
                'union__txt4__tsvd4__n_components': [ncomp_svd]
                }
    
    xfrm = FeatureUnion(
        transformer_list = [
            ('cst',  cust_regression_vals()),  
            ('txt1', pipeline.Pipeline([('s1', cust_txt_col(key='search_term_segs_in_title')), ('tfidf1', tfidf), ('tsvd1', tsvd)])),
            ('txt2', pipeline.Pipeline([('s2', cust_txt_col(key='product_title')), ('tfidf2', tfidf), ('tsvd2', tsvd)])),
            ('txt3', pipeline.Pipeline([('s3', cust_txt_col(key='product_description')), ('tfidf3', tfidf), ('tsvd3', tsvd)])),
            ('txt4', pipeline.Pipeline([('s4', cust_txt_col(key='brand')), ('tfidf4', tfidf), ('tsvd4', tsvd)]))
            ],
        n_jobs = 1
        )
    
    clf = pipeline.Pipeline([
            ('union', xfrm), 
            ('rgr', rgr)])
    
    # grid search with ten fold cross validation
    if cv == -1:
        # in this case we xfrm the data and then run rgr on it directly, fro additive models it helps to plot n_estimators vs
        # fit at each step
        xfrm.set_params(**xparam_grid)
        X_train_xd = xfrm.fit_transform(X_train)
        X_model_test_xd = xfrm.fit_transform(X_model_test)
        X_test_xd = xfrm.fit_transform(X_test)
        train_df_xd = xfrm.fit_transform(train_df)        
        save_model_results(
            X_train_xd,
            y_train,
            X_model_test_xd, 
            y_model_test,            
            train_df_xd,
            train_df['relevance'],
            X_test_xd,
            mstring,
            rgr)
    elif cv > 1:
        # grid search with cv
        print("Fitting model for "+str(i))
        model = grid_search.GridSearchCV(estimator = clf, param_grid = param_grid, n_jobs = -1, cv = cv, verbose = 20, scoring=RMSE,error_score=0)
        save_model_results(
            X_train,
            y_train,
            X_model_test, 
            y_model_test,
            train_df,
            train_df['relevance'],
            X_test,
            mstring,
            clf,
            model)
    else:
        # this is no grid search just a pipeline
        clf = clf.set_params(**param_grid)
        save_model_results(
            X_train,
            y_train,
            X_model_test, 
            y_model_test,
            train_df,
            train_df['relevance'],          
            X_test,
            mstring,
            clf)

        

#-------------------------------------------------------------------------
#  save the trained cv results for plotting
#-------------------------------------------------------------------------
#joblib.dump(model, 'model_pkl/model_xgbr_n1000_vgamma_d5_vlr.pkl') 
#model = joblib.load('model_pkl/model_xgbr_n1000_vgamma_d5_vlr.pkl') 

#joblib.dump(model, 'model_pkl/model_xgbr_n1000_vgamma_d10_vlr.pkl') 
#model = joblib.load('model_pkl/model_xgbr_n1000_vgamma_d10_vlr.pkl') 

#joblib.dump(model,'model_pkl/model_xgbr_vn_gamma05_lr001_d5_vlr.pkl') 
#model = joblib.load('model_pkl/model_xgbr_vn_gamma05_lr001_d5_vlr.pkl') 
    
#joblib.dump(model,'model_pkl/model_xgbr_vn_vgamma_vlr001_d5.pkl') 
#model = joblib.load('model_pkl/model_xgbr_vn_vgamma_vlr001_d5.pkl') 


#plot the 3D for grid scores
gslist = []
for p, f1m, f1s in model.grid_scores_:
    k = p
    k['rmse'] = np.abs(f1m)
    k['std'] = f1s.std()
    gslist.append(k)

gsdf = pd.DataFrame(gslist)

# columns to plot
xcol2 = 'rgr__learning_rate'
ycol = 'rgr__gamma'
zcol = 'rmse'

xcol = 'rgr__n_estimators'

cols_of_interest = [xcol,ycol,zcol,xcol2]

# filter the 3 columsn to plot
gs_uv = gsdf[xcol2].unique()

plt.close('all')

fig = plt.figure(figsize=(12, 10))

ax = fig.add_subplot(111, projection='3d')

gs_col = ['r','b','g','k','m']
gs_m = ['o','^','8','*','s']

gs_lg_proxy = []

for i in range(len(gs_uv)):
    gs_plt_df = gsdf[gsdf[xcol2]==gs_uv[i]].loc[:,cols_of_interest]    
    #ax.plot_surface(gs_plt_df[xcol],gs_plt_df[ycol],gs_plt_df[zcol],rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    ax.scatter(gs_plt_df[xcol],gs_plt_df[ycol],gs_plt_df[zcol],c=gs_col[i], marker=gs_m[i], s=15)
    lg_prxy = plt.Rectangle((0, 0), 1, 1, fc=gs_col[i])
    gs_lg_proxy.append(lg_prxy)

    #ax.plot_wireframe(gs_plt_df[xcol],gs_plt_df[ycol],gs_plt_df[zcol],rstride=100, cstride=100)
    #ax.contour(gs_plt_df[xcol],gs_plt_df[ycol],gs_plt_df[zcol])

ax.legend(gs_lg_proxy,gs_uv)
ax.set_xlabel('Estimators')
ax.set_ylabel('gamma')
ax.set_zlabel('RMSE')

ax.set_title('RMSE vs No. Estimators and gamma for different Learning Rates')

fig.suptitle('Cross Validation Plot for Extreme Gradient Boosting Regressor Tree Depth = 5')
    
fig.savefig('3dplot_xgbr.png', bbox_inches='tight')

plt.show()

plt.close('all')

#  Best Parameters from xgbr
#{'rgr__gamma': 0,
# 'rgr__learning_rate': 0.01,
# 'rgr__max_depth': 5,
# 'rgr__n_estimators': 600,
# 'rgr__subsample': 1.0,
# 'union__transformer_weights': [1.0, 0.75, 0.5, 0.15, 1],
# 'union__txt1__tfidf1__ngram_range': (1, 3),
# 'union__txt1__tsvd1__n_components': 10,
# 'union__txt2__tfidf2__ngram_range': (1, 3),
# 'union__txt2__tsvd2__n_components': 10,
# 'union__txt3__tfidf3__ngram_range': (1, 3),
# 'union__txt3__tsvd3__n_components': 10,
# 'union__txt4__tfidf4__ngram_range': (1, 3),
# 'union__txt4__tsvd4__n_components': 10} }

# warm fitting the gradientboostregressor with warm start - custom execution
# clf.set_params(**{'rgr__n_estimators' : 800, 'rgr__warm_start': True})

# as no of estimators go up a smaller learning rate gives better results
#-----------------------------------------------------------------------
# save the all_results
#-----------------------------------------------------------------------
ts = str(int(time.time()))

with open('data_pkl/' +  ts + 'all_results.json', 'w') as f:
    f.write(json.dumps(all_results))


#-----------------------------------------------------------------------
# Analyse results of all runs so far
#-----------------------------------------------------------------------
# from all_results sort
#'rmse_train_refit'
#'rmse_test_data'
#'rmse_best_cv_score'
#'cv_best_params'

# read the last all_results file from dir and save that as all_results
all_result_files = [file for file in glob.glob(os.path.join(*[os.getcwd(), 'data_pkl', '*all_results.json']))]
if all_result_files:
    all_result_files.sort(key=os.path.getmtime)
    with open(all_result_files[-1]) as all_results_latest:
        all_results = json.load(all_results_latest)

sorted_rmse_test_data = sorted(all_results['rmse_test_data'].items(), key=operator.itemgetter(1))
sorted_rmse_train_refit = sorted(all_results['rmse_train_refit'].items(), key=operator.itemgetter(1))
sorted_rmse_best_cv_score = sorted(all_results['rmse_best_cv_score'].iteritems(), key=lambda (k,v): (np.abs(v),k))
sorted_rmse_test_data_closest = sorted(all_results['rmse_test_data_closest'].items(), key=operator.itemgetter(1))
sorted_rmse_train_refit_closest = sorted(all_results['rmse_train_refit_closest'].items(), key=operator.itemgetter(1))

# to check if there is a bug in rmse_test_data
#pp.pprint(set(all_results['rmse_train_refit'].keys()) - set(all_results['rmse_test_data'].keys()))

dict1 = [dict(sorted_rmse_test_data) , dict(sorted_rmse_train_refit), dict(sorted_rmse_best_cv_score) ]

dictf = {}
for d in dict1:
    for k, v in d.iteritems():
        if k in dictf:
            dictf[k] = dictf[k] + np.abs(v)
        else:
            dictf[k] = np.abs(v) 

     
dictf = { k : v/3 for k,v in dictf.items() }
sorted_dictf = sorted(dictf.items(), key=operator.itemgetter(1))

print("sorted_rmse_best_cv_score")
pp.pprint(sorted_rmse_best_cv_score[:10])

print(os.linesep+"sorted_rmse_train_refit")
pp.pprint(sorted_rmse_train_refit[:10])

print(os.linesep+"sorted_rmse_train_refit_closest")
pp.pprint(sorted_rmse_train_refit_closest[:10])

print(os.linesep+"sorted_rmse_test_data")
pp.pprint(sorted_rmse_test_data[:10])

print(os.linesep+"sorted_rmse_test_data_closest")
pp.pprint(sorted_rmse_test_data_closest[:10])

top_k = 5

print(os.linesep+"Mean of all the top "+str(top_k)+" test scores")
pp.pprint(sorted_dictf[:top_k])

print(os.linesep+"Parameters of the top "+str(top_k))
for v in sorted_dictf[:top_k]:
    print("Results for "+v[0])
    for k in ('rmse_test_data', 'rmse_test_data_closest', 'rmse_train_refit', 'rmse_train_refit_closest', 'rmse_best_cv_score' ):
        if v[0] in all_results[k]:
            print( k + " " + str(all_results[k][v[0]]))
        else:
            print( k + " ")
    if v[0] in all_results:
        print("random seed " + str(all_results[v[0]]['randomseed']))
    else:
        print("random seed " + str(crandomseed))
    if v[0] in all_results['cv_best_params']:
        pp.pprint(all_results['cv_best_params'][v[0]])
    print(os.linesep)



#-------------------------------------------------------------------------
#  save the trained classed
#-------------------------------------------------------------------------
joblib.dump(laso, 'model_pkl/laso_model.pkl') 
joblib.dump(rfr, 'model_pkl/rfr_model.pkl') 
joblib.dump(gbr, 'model_pkl/gbr_model.pkl') 
joblib.dump(xgbr, 'model_pkl/xgbr_model.pkl') 

laso = joblib.load('model_pkl/laso_model.pkl') 
rfr = joblib.load('model_pkl/rfr_model.pkl') 
gbr = joblib.load('model_pkl/gbr_model.pkl') 
xgbr = joblib.load('model_pkl/xgbr_model.pkl') 


#-------------------------------------------------------------------------
# Bootstrapping the 4 models to compare and plot results
#-------------------------------------------------------------------------
bs_iter = 30
bs_test_frac = 0.40

shfl = bst.ShuffleSplit(train_df_xd.shape[0], 
                     n_iter=bs_iter,
                     test_size=bs_test_frac, 
                     random_state=randomseed)

bs_train_error = pd.DataFrame(columns=('Lasso', 'RandomForestRegressor', 'GradientBoostingRegressor', 'XGBRegressor'))
bs_test_error = pd.DataFrame(columns=('Lasso', 'RandomForestRegressor', 'GradientBoostingRegressor', 'XGBRegressor'))

crfr = clone(rfr)
claso = clone(laso)
cgbr = clone(gbr)
cxgbr = clone(xgbr)

# set the best properties for n_estimators seen for each model
crfr.set_params(n_estimators=300)
cgbr.set_params(n_estimators=300)
cxgbr.set_params(n_estimators=300)
rtype=None
for train_index, test_index in shfl:
    X_bs_train_xd = train_df_xd[train_index]
    y_bs_train = np.array(train_df['relevance'].values[train_index])
    X_bs_test_xd = train_df_xd[test_index]
    y_bs_test = np.array(train_df['relevance'].values[test_index])
    
    pred_train = {}
    pred_test = {}
    for est in [ crfr, claso, cgbr, cxgbr ]:
        
        # around n_estimators give good results for bootstrapping
        if 'Lasso' in  str(type(est)):
            rtype = 'Lasso'
        elif 'RandomForestRegressor' in str(type(est)):
            rtype = 'RandomForestRegressor'            
        elif 'GradientBoostingRegressor' in str(type(est)):
            rtype = 'GradientBoostingRegressor'
        elif 'XGBRegressor' in str(type(est)):
            rtype = 'XGBRegressor'
        
        est.fit(X_bs_train_xd,y_bs_train)
        y_bs_train_pred = est.predict(X_bs_train_xd)    
        y_bs_test_pred = est.predict(X_bs_test_xd)
        bs_train_rmse = fmean_squared_error(y_bs_train, y_bs_train_pred)
        bs_test_rmse = fmean_squared_error(y_bs_test, y_bs_test_pred)
        
        pred_train[rtype] = bs_train_rmse
        pred_test[rtype] = bs_test_rmse
    
    bs_train_error.loc[len(bs_train_error)] = pred_train
    bs_test_error.loc[len(bs_test_error)] = pred_test


# save to file
ts = str(int(time.time()))

bs_train_error.to_pickle('data_pkl/'+ts+'bs_train_error.pkl')
bs_test_error.to_pickle('data_pkl/'+ts+'bs_test_error.pkl')

# read the last all_results file from dir and save that as all_results
bs_train_error_files = [file for file in glob.glob(os.path.join(*[os.getcwd(), 'data_pkl', '*bs_train_error.pkl']))]
if bs_train_error_files:
    bs_train_error_files.sort(key=os.path.getmtime)
    bs_train_error = pd.read_pickle(bs_train_error_files[-1])

bs_test_error_files = [file for file in glob.glob(os.path.join(*[os.getcwd(), 'data_pkl', '*bs_test_error.pkl']))]
if bs_test_error_files:
    bs_test_error_files.sort(key=os.path.getmtime)
    bs_test_error = pd.read_pickle(bs_test_error_files[-1])

# box plot of model performance
plt.close('all')
   
fig, gph = plt.subplots(nrows=2, sharex=True, figsize=(10, 12))

bs_train_error.boxplot(ax=gph[0])
bs_test_error.boxplot(ax=gph[1])

gph[0].set_ylabel('Error')
gph[1].set_ylabel('Error')

xtickNames = gph[1].xaxis.set_ticklabels(bs_test_error.columns.values,rotation=45)
plt.setp(xtickNames, rotation=30, fontsize=11)

gph[0].set_title('Training Error rates and Standard Deviation')
gph[1].set_title('Testing Error rates and Standard Deviation')
fig.suptitle('Comparing Models Used - Testing Error rates and Standard Deviation')
    
fig.savefig('models.png', bbox_inches='tight')
plt.show()
plt.close()

plt.close('all')


#-------------------------------------------------------------------------
# DO a t test and a wilcox test cmparing bootstrap results to xgbr
#-------------------------------------------------------------------------
comp_stats = {}
comp_stats['t_stats'] = {}
comp_stats['p_vals'] = {}
comp_stats['w_stats'] = {}
comp_stats['p_w_vals'] = {}
for col in bs_test_error.columns:
    if col != 'XGBRegressor':
        t_stat, p_val = stats.ttest_ind(bs_test_error['XGBRegressor'],bs_test_error[col], equal_var=False)
        comp_stats['t_stats'][col] = t_stat
        comp_stats['p_vals'][col] = p_val
        t_stat, p_val = stats.wilcoxon(bs_test_error['XGBRegressor'],bs_test_error[col])
        comp_stats['w_stats'][col] = t_stat
        comp_stats['p_w_vals'][col] = p_val

print("t test and wilcox test results comparing with bootstrapping results from XGBRegressor")
pp.pprint(comp_stats)

# save to file
ts = str(int(time.time()))

# save the test errors to file
with open('data_pkl/'+ts+'bs_t_test.pkl', 'wb+') as handle:
  pickle.dump(comp_stats, handle)

# read the last all_results file from dir and save that as all_results
bs_t_test_files = [file for file in glob.glob(os.path.join(*[os.getcwd(), 'data_pkl', '*bs_t_test.pkl']))]
if bs_t_test_files:
    bs_t_test_files.sort(key=os.path.getmtime)
    with open(bs_t_test_files[-1], 'rb') as handle:
        comp_stats = pickle.load(handle)


#--------------------------------------------------------------------------
# Estimator performance analysis
#--------------------------------------------------------------------------
# WE need to first transform the data in the pipeline before using the stagewise prediction
p = []
def estimator_perf(est):
    #est = ppl.steps[1][1]
    for i,tree in enumerate(est.estimators_):
        print(i)
        y_train_pred = tree.staged_predict(X_test_xd)
        y_train_refit_rmse = fmean_squared_error(train_df['relevance'].values, y_train_pred)
        p.insert(i,y_train_refit_rmse)
        


#----------------------------------------------------------------------------
# Plot training deviance

# compute test set deviance
# GradientBoostingRegressor
#-----------------------------------------------------------------------------
# compute test set deviance
#>>> xgbr.get_params()
#{'reg_alpha': 0, 'colsample_bytree': 1, 'silent': False, 'colsample_bylevel': 1, 'scale_pos_weight': 1, 'learning_rate': 0.01, 'missing': None, 'max_delta_step': 0, 'nthread': 1, 'base_score': 0.5, 'n_estimators': 1000, 'subsample': 1.0, 'reg_lambda': 1, 'seed': 0, 'min_child_weight': 1, 'objective': 'reg:linear', 'max_depth': 5, 'gamma': 0}

plt.close('all')
for est in [ gbr ]:
    
    t_estimators = 1000
    est_test_score = np.zeros(t_estimators, dtype=np.float64)
    est_train_score = np.zeros(t_estimators, dtype=np.float64)        
    est_train_m_score = np.zeros(t_estimators, dtype=np.float64)   
    
    if 'RandomForestRegressor' in str(type(est)):
        title = 'Random Forest Regressor'
        fname = 'rfr_ed'  
        params = est.get_params()
        est = RandomForestRegressor()
        est.set_params(**params)        
        est.set_params(**{ 'warm_start':True, 'oob_score':True })
        for i in range(t_estimators):            
            est.set_params(n_estimators=i+1)
            est.fit(X_train_xd,y_train)
            oob_error = 1 - est.oob_score_
            y_s_pred = est.predict(X_model_test_xd)
            est_test_score[i] = fmean_squared_error(y_model_test, y_s_pred)
            y_s_pred = est.predict(X_train_xd)
            est_train_score[i] = fmean_squared_error(y_train, y_s_pred)      
    elif 'GradientBoostingRegressor' in str(type(est)):
        title = 'Gradient Boosting Regressor'
        fname = 'gbr_ed_n'
        # make sure we train the model again on the train and test data
        #  we we can get the test and train error
        est.set_params(n_estimators=t_estimators)
        est.fit(X_train_xd,y_train)  
        # The train error at each iteration is stored in the train_score_ attribute 
        # of the gradient boosting model. 
        # The test error at each iterations can be obtained via the staged_predict 
        # method which returns a generator that yields the predictions at each stage
        for i, y_s_pred in enumerate(est.staged_predict(X_model_test_xd)):            
            est_test_score[i] = est.loss_(y_model_test, y_s_pred)
        #for i, y_s_pred in enumerate(est.staged_predict(X_train_xd)):
        #    est_train_score[i] = fmean_squared_error(y_train, y_s_pred)
        est_train_score = est.train_score_
    elif 'XGBRegressor' in str(type(est)):
        title = 'Extreme Gradient Boosting Regressor'
        fname = 'xgbr_ed_n'
        # make sure we train the model again on the train and test data
        #  we we can get the test and train error
        est.set_params(n_estimators=t_estimators)
        est.fit(X_train_xd,y_train)    
        for i in range(t_estimators):  
            # we have to do this thru xgbboost as 
            # XGBRegressor does not have a way to predic using the first n trees
            y_s_pred = est.booster().predict(xgb.DMatrix(X_model_test_xd), ntree_limit=i+1)            
            est_test_score[i] = fmean_squared_error(y_model_test, y_s_pred)
            y_s_pred = est.booster().predict(xgb.DMatrix(X_train_xd), ntree_limit=i+1)
            est_train_score[i] = fmean_squared_error(y_train, y_s_pred)
    
    plt.figure(figsize=(12, 10))    
    
    plt.plot(np.arange(t_estimators) + 1, est_train_score, 'b-',
             label='Training Set Deviance')
    
    plt.plot(np.arange(t_estimators) + 1, 
             est_test_score, 'r-',
             label='Test Set Deviance')
    
    plt.legend(loc='upper right')
        
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')
    plt.title(title + ' - Test and Training Deviance')
    
    fig = plt.gcf()
    fig.savefig(fname+'.png', bbox_inches='tight')
    
    plt.show()
    plt.close()
    
    np.savetxt('data_pkl/'+fname +'_est_train_score.csv', est_train_score, delimiter=",")
    np.savetxt('data_pkl/'+fname +'_est_test_score.csv', est_test_score, delimiter=",")
    joblib.dump(est, 'model_pkl/'+fname+'_model.pkl') 

plt.close('all')

#-----------------------------------------------------------------------------
# Plot feature importance
# GradientBoostingRegressor and RandomForestRegressor
#-----------------------------------------------------------------------------
plt.close('all')
for est in [rfr, gbr, xgbr]:
    
    if 'RandomForestRegressor' in str(type(est)):
        title = 'Random Forest Regressor'
        fname = 'rfr_fp.png'
        feature_importance = est.feature_importances_
    elif 'GradientBoostingRegressor' in str(type(est)):
        title = 'Gradient Boosting Regressor'
        fname = 'gbr_fp.png'
        feature_importance = est.feature_importances_
    elif 'XGBRegressor' in str(type(est)):
        title = 'Extreme Gradient Boosting Regressor'
        fname = 'xgbr_fp.png'
        pd.Series(est.booster().get_fscore()).sort_values(ascending=False)
    
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    
    # take top 20 for plotting    
    sorted_idx = np.argsort(feature_importance)[::-1][:20][::-1]
    pos = np.arange(sorted_idx.shape[0]) + .5    
    plt.figure(figsize=(12, 10))    
    plt.barh(pos, feature_importance[sorted_idx], align='center')    
    plt.yticks(pos, feature_names[sorted_idx])    
    plt.xlabel('Relative Importance (0-100%)')    
    plt.title(title + ' - Feature Importance')
    
    fig = plt.gcf()
    fig.savefig(fname, bbox_inches='tight')
    plt.show()
    plt.close()

plt.close('all')

exit()
