# This Python file uses the following encoding: utf-8
#
# ajdsouza31 - dl
#
# Submitted as final project work project ISYE-7406Q
#
# Kaggle Home Depot competition
#
# This file reads the data and creates a feature vector and saves it to disk
#

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
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from cycler import cycler


from nltk.stem.porter import *
stemmer = PorterStemmer()
#from nltk.stem.snowball import SnowballStemmer #0.003 improvement but takes twice as long as PorterStemmer
#stemmer = SnowballStemmer('english')

import random
random.seed(20160417)

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

#---------------------------------------------------------------------
# Support functions
#---------------------------------------------------------------------

# list of stop words
stop_w = ['for', 'xbi', 'and', 'in', 'th','on','sku','with','what','from','that','less','er','ing']

# conversion of words to digits
strNum = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9}

# the first stemmer for words
def str_stem(s):
    if not s:
        return "null"
    # remove non ascii characters
    s = re.sub(r'[^\x00-\x7F]+',' ', s)
    # convert to string
    if not s:
        return "null"
    s = str(s)
    if isinstance(s, str):
        s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s) #Split words with a.A
        s = s.lower()
        s = s.replace("  "," ")
        s = s.replace(",","") #could be number / segment later
        s = s.replace("$"," ")
        s = s.replace("?"," ")
        s = s.replace("-"," ")
        s = s.replace("//","/")
        s = s.replace("..",".")
        s = s.replace(" / "," ")
        s = s.replace(" \\ "," ")
        s = s.replace("."," . ")
        s = re.sub(r"(^\.|/)", r"", s)
        s = re.sub(r"(\.|/)$", r"", s)
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
        s = s.replace(" x "," xbi ")
        s = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", s)
        s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)
        s = s.replace("*"," xbi ")
        s = s.replace(" by "," xbi ")
        s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
        s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
        s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
        s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
        s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)
        s = s.replace("°"," degrees ")
        s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)
        s = s.replace(" v "," volts ")
        s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)
        s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)
        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)
        s = s.replace("  "," ")
        s = s.replace(" . "," ")
        #s = (" ").join([z for z in s.split(" ") if z not in stop_w])
        s = (" ").join([str(strNum[z]) if z in strNum else z for z in s.split(" ")])
        s = (" ").join([stemmer.stem(z) for z in s.split(" ")])
        
        s = s.lower()
        s = s.replace("toliet","toilet")
        s = s.replace("airconditioner","air conditioner")
        s = s.replace("vinal","vinyl")
        s = s.replace("vynal","vinyl")
        s = s.replace("skill","skil")
        s = s.replace("snowbl","snow bl")
        s = s.replace("plexigla","plexi gla")
        s = s.replace("rustoleum","rust-oleum")
        s = s.replace("whirpool","whirlpool")
        s = s.replace("whirlpoolga", "whirlpool ga")
        s = s.replace("whirlpoolstainless","whirlpool stainless")
        return s
    else:
        return "null"
    


# count of segments of each of the words in search string str1 that are in words in the str2
def seg_words(str1, str2):
    # lower case the string to be searched in
    str2 = str2.lower()
    # remove non alpha num in str2, the string to be searched in
    str2 = re.sub("[^a-z0-9./]"," ", str2)
    # remove words <= 2 from the string to be searched in and make it a 
    # list
    str2 = [z for z in set(str2.split()) if len(z)>2]
    # change search string to lower and convert it into a word array
    words = str1.lower().split(" ")
    s = []
    # for each word in the search string 
    # if the word is <= 3 just add it to the s[]
    # for words > 3 
    # find the continuous segments of the word matching the words in strings to be searched
    for word in words:
        if len(word)>3:
            s1 = []
            s1 += segmentit(word,str2,True)
            # if segments of word match words in str2 then add those segments to s
            # shouldnt this be len(s1) instead of len(s) ?
            # I changed this not sure if it will hold
            if len(s1)>1:
                s += [z for z in s1 if z not in ['er','ing','s','less'] and len(z)>1]
            else:
                s.append(word)
        else:
            s.append(word)
    # append all those segments in search that match the text to be search and send back the string
    return (" ".join(s))




# list of the various segments of a single word s starting left to right 
# that match in any word in text_arr
# The segment should be continuous form the begining of s
#
# When part of s from begining matches a word in txt_arr , that part of s is added to the
# return array and the search is continued. Sp if a part fo the word matches multiple words in 
# txt_arr only the first match is taken
#
# it returns a list of continous segments from left to right that match
# words in text_arr, with the remainder of the unmatched segment from s as the 
# last element in the array 
def segmentit(s, txt_arr, t):
    st = s
    r = []
    # print("invoked segment it with string "+s+" and text array "+str(txt_arr))
    # from 0..len(s)-1
    for j in range(len(s)):
        # for each word in text_arr
        #print("j is %s, s is %s" % (j, s))
        for word in txt_arr:
            # if the word from text_arr matches the the string from begining till j from the end
            #  word == s[0:len(s)-j]  ie word  == s[0....len(s)-j]
            if word == s[:-j]:
                # append  the matched substring s[0....len(s)-j] to list r
                r.append(s[:-j])
                #print("matched %s remaining %s" % (s[:-j],s[len(s)-j:]))
                # now take the remaining of the substring s ie s[len(s)-j...] as the new s
                s=s[len(s)-j:]
                # print("the new s is %s" % s)
                # run a segmentit on this remainig s to get the segments of strings in s that
                # match words in txt_attr
                r += segmentit(s, txt_arr, False)
                # further continuation of the loop beyond this is a dying loop that will not
                # add any segments in the outer loop as counter j now moves beyond the 
                # length of the string and will return a null for s[j:] and s[:-j]
    # true only for the top most call
    #  if segment of s in txt_att is less than the length of s
    #  then add the balance of s to r
    if t:
        i = len(("").join(r))
        if not i==len(st):
            r.append(st[i:])
    return r


# count of the words in str1 that exist in str2
# make everythign lowercase before comparing
def str_common_word(str1, str2):
    words, cnt = str1.lower().split(), 0
    for word in words:
        if str2.lower().find(word)>=0:
            cnt+=1
    return cnt


# count of # of times str1 is in str2
# make everything lowercase before comparing
def str_whole_ngram(str1, str2, i_):
    cnt = 0
    while i_ < len(str2):
        i_ = str2.lower().find(str1.lower(), i_)
        if i_ == -1:
            return cnt
        else:
            cnt += 1
            i_ += len(str1)
    return cnt


# JAccard coefficient between 2 strings
def JaccardCoefficient(str1, str2):
    st1 = set(str1.lower().split())
    st2 = set(str2.lower().split())
    return float(len(st1 & st2)) / len(st1 | st2)

#---------------------------------------------------------------------
# read data
#---------------------------------------------------------------------
# read the train,test data from csv files
train_df = pd.read_csv('../kaggle-homedepot/train.csv/train.csv', encoding="ISO-8859-1")
test_df = pd.read_csv('../kaggle-homedepot/test.csv/test.csv', encoding="ISO-8859-1")
product_desc_df = pd.read_csv('../kaggle-homedepot/product_descriptions.csv/product_descriptions.csv')
product_attr_df = pd.read_csv('../kaggle-homedepot/attributes.csv/attributes.csv')

# read the save search string spell corrections
spell_checker_df = pd.read_csv('spell_checked_search.csv', encoding="ISO-8859-1")
spell_checker = spell_checker_df.set_index('search')['spell_corrected'].to_dict()

print(">>>>>>>>>Time to read the data %d" % round(((time.time() - time0)/60),2))


#--------------------------------------------------------------------
# Data clean up
#--------------------------------------------------------------------
# remove the null product ids in attrib
product_attr_df = product_attr_df[product_attr_df.product_uid.notnull()]

# remove outlier data for relevance values in train set


# concateante the train and test dataframes so we can generate feature vectors in one go
# keep count of the train data size
train_count = train_df.shape[0]
all_data_df = pd.concat((train_df,test_df),axis=0,ignore_index=True)

# correct the search spell using spell checker
all_data_df["search_term"].replace(spell_checker, inplace=True)

print(">>>>>>>>>Time to clean up the data %d" % round(((time.time() - time0)/60),2))


#-----------------------------------------------------------------------
#  Build the feature vectors
#-----------------------------------------------------------------------

#-------------------------------------------------------------------------------------
# 1. extract the brand name from product attributes for each product
# 2. then merge brand name column to the all data df and product description
#       based on product_uid
#-------------------------------------------------------------------------------------
# exploratory analysis shows
#  1.  brand name is in attribute "mfg brand name"
#
# build a data frame of productuid,brand based on atte "mfg brand name" attrib
brandre = re.compile(r'mfg brand name',re.IGNORECASE)
brand_name_df = product_attr_df[product_attr_df.name.str.contains(brandre,na=False)][["product_uid", "value"]].rename(columns={"value": "brand"})
brand_name_df['product_uid'] = brand_name_df['product_uid'].astype(int)
all_data_df = pd.merge(all_data_df, product_desc_df, how='left', on='product_uid')
all_data_df = pd.merge(all_data_df, brand_name_df, how='left', on='product_uid')

# convert some of the 
#all_data_df['search_term'] = all_data_df['search_term'].astype(str)
#all_data_df['product_title'] = all_data_df['product_title'].astype(str)
#all_data_df['product_description'] = all_data_df['product_description'].astype(str)
all_data_df['brand'] = all_data_df['brand'].astype(str)



#import sys
#reload(sys)
#sys.setdefaultencoding("utf-8")
#-------------------------------------------------------------------------------------
# Stem the text fields in search, title, description and brand
#  in the all data df
#-------------------------------------------------------------------------------------
all_data_df['search_term'] = all_data_df['search_term'].map(lambda x:str_stem(x))
all_data_df['product_title'] = all_data_df['product_title'].map(lambda x:str_stem(x))
all_data_df['product_description'] = all_data_df['product_description'].map(lambda x:str_stem(x))
all_data_df['brand'] = all_data_df['brand'].map(lambda x:str_stem(x))
print(">>>>>>>>>Stemming: %s minutes ---" % round(((time.time() - time0)/60),2))

#-------------------------------------------------------------------------------------
#  1. get the number of words in search, title,description and brand
#-------------------------------------------------------------------------------------
# count of the terms in search term - a plot of the relevance vs count does not show much of a relationship
all_data_df['len_of_query'] = all_data_df['search_term'].map(lambda x:len(x.split())).astype(np.int64)
all_data_df['len_of_title'] = all_data_df['product_title'].map(lambda x:len(x.split())).astype(np.int64)
all_data_df['len_of_description'] = all_data_df['product_description'].map(lambda x:len(x.split())).astype(np.int64)
all_data_df['len_of_brand'] = all_data_df['brand'].map(lambda x:len(x.split())).astype(np.int64)
print(">>>>>>>>>Len of: %s minutes ---" % round(((time.time() - time0)/60),2))


#-------------------------------------------------------------------------------------------
#  1. add a new field product_info which is search+title+desctiption delimited by new line
#-------------------------------------------------------------------------------------------
all_data_df['product_info'] = all_data_df['search_term']+"\t"+all_data_df['product_title'] +"\t"+all_data_df['product_description']
print(">>>>>>>>>>Prod Info: %s minutes ---" % round(((time.time() - time0)/60),2))

#------------------------------------------------------------------------------------------
#  search_term -> all segments of search term in product_title
#------------------------------------------------------------------------------------------
all_data_df['search_term_segs_in_title'] = all_data_df['product_info'].map(lambda x:seg_words(x.split('\t')[0],x.split('\t')[1]))
print(">>>>>>>>>>>>Search Term Segment: %s minutes ---" % round(((time.time() - time0)/60),2))

#--------------------------------------------------------------------------------------------
#  query_in_title, query_in_description
#   -> count of whole search string in title, 
#   -> count of whole search string in description
#--------------------------------------------------------------------------------------------
all_data_df['query_in_title'] = all_data_df['product_info'].map(lambda x:str_whole_ngram(x.split('\t')[0],x.split('\t')[1],0))
all_data_df['query_in_description'] = all_data_df['product_info'].map(lambda x:str_whole_ngram(x.split('\t')[0],x.split('\t')[2],0))
print(">>>>>>>>>Query In: %s minutes ---" % round(((time.time() - time0)/60),2))

#------------------------------------------------------------------------------------------
#  query_last_word_in_title,query_last_word_in_description
#    -> last word from search in title, description ?
#------------------------------------------------------------------------------------------
all_data_df['query_last_word_in_title'] = all_data_df['product_info'].map(lambda x:str_common_word(x.split('\t')[0].split(" ")[-1],x.split('\t')[1]))
all_data_df['query_last_word_in_description'] = all_data_df['product_info'].map(lambda x:str_common_word(x.split('\t')[0].split(" ")[-1],x.split('\t')[2]))
print(">>>>>>>>> Query Last Word In: %s minutes ---" % round(((time.time() - time0)/60),2))

#------------------------------------------------------------------------------------------
#  word_in_title,word_in_description
#    -> count of words from search in title
#    -> count of the words from search in description
#------------------------------------------------------------------------------------------
all_data_df['word_in_title'] = all_data_df['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
all_data_df['word_in_description'] = all_data_df['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))

#------------------------------------------------------------------------------------------
#  ratio_title,ratio_description
#    -> count of words from search in title / # of words in search
#    -> count of words from search in description / # of words in search
#------------------------------------------------------------------------------------------
all_data_df['ratio_title'] = all_data_df['word_in_title']/all_data_df['len_of_query']
all_data_df['ratio_description'] = all_data_df['word_in_description']/all_data_df['len_of_query']


#------------------------------------------------------------------------------------------
#  word_in_brand, ratio_brand, ratio_brand_brand
#    -> count of words from search segs in brand
#    -> count of words from search segs in brand / # of words in search
#    -> count of words from search segs in brand / # of words in brand
#------------------------------------------------------------------------------------------
all_data_df['attr'] = all_data_df['search_term_segs_in_title']+"\t"+all_data_df['brand']
all_data_df['word_in_brand'] = all_data_df['attr'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
all_data_df['ratio_brand'] = all_data_df['word_in_brand']/all_data_df['len_of_query']
all_data_df['ratio_brand_brand'] = all_data_df['word_in_brand']/all_data_df['len_of_brand']

#------------------------------------------------------------------------------------------
#  search_segment_feature, brand_feature
#    -> count of words in search segs
#    -> a numeric id for each unique brand
#------------------------------------------------------------------------------------------
all_data_df['search_term_feature'] = all_data_df['search_term_segs_in_title'].map(lambda x:len(x))

df_brand_unique = pd.unique(all_data_df.brand.ravel()) 
d={}
i = 1000
for s in df_brand_unique:
    d[s]=i
    i+=3

all_data_df['brand_feature'] = all_data_df['brand'].map(lambda x:d[x])



#------------------------------------------------------------------------------------------
# frequency of a product uid, relevance
#------------------------------------------------------------------------------------------
#The average relevance score is examined by nummber of times a product_uid occurs in the dataset.
# variance in relevance score decreases as the count of product uid increases
pid_frequency = dict(all_data_df.groupby('product_uid')['product_uid'].count()*1000000/all_data_df.shape[0])
all_data_df['pid_count'] = all_data_df['product_uid'].map(lambda x: pid_frequency[x] )


# frequency plot of the relevance scores
# well rounded scores - perhaps not more than 1 evaluated relevance
relv_frequency = dict(train_df.groupby('relevance')['relevance'].count())

# remove relevance scores with count < 20 cont 59 approx 0.0797% , they are 
# in the outlier range
relvals = [k for k, v in relv_frequency.items() if v >= 20]

#jaccard coefficients search and title
all_data_df['jaccard_search_title'] = all_data_df['product_info'].map(lambda x:JaccardCoefficient(x.split('\t')[0],x.split('\t')[1]))

#jaccard coefficients search and description
all_data_df['jaccard_search_description'] = all_data_df['product_info'].map(lambda x:JaccardCoefficient(x.split('\t')[0],x.split('\t')[2]))

#jaccard coefficients search and brand
all_data_df['jaccard_search_brand'] = all_data_df.apply(lambda row: JaccardCoefficient(row['search_term'],row['brand']),axis=1)



print(">>>>>>>>>Time to build feature vectors %d" % round(((time.time() - time0)/60),2))


#---------------------------------------------------------------------------------------
# save the feature vector train and test data to csv file
#---------------------------------------------------------------------------------------
train_df = all_data_df.iloc[:train_count]
test_df = all_data_df.iloc[train_count:]

all_data_df.to_csv('all_data_df.csv')
train_df.to_csv('train_df.csv')
test_df.to_csv('test_df.csv')


#----------------------------------------------------------------------------
# PLOTS BEGN HERE
#----------------------------------------------------------------------------

#---------------------------------------------------------------------------------------
# Data exploration and Analysis
#---------------------------------------------------------------------------------------
train_df = pd.read_csv('train_df.csv', encoding="ISO-8859-1", index_col=0)
test_df = pd.read_csv('test_df.csv', encoding="ISO-8859-1", index_col=0)

# concateante the train and test dataframes so we can generate feature vectors in one go
# keep count of the train data size
train_count = train_df.shape[0]
all_data_df = pd.concat((train_df,test_df),axis=0,ignore_index=True)


# remove rows which have relevance scores count > 2, 59/74K outliers <.001%
relv_frequency = dict(train_df.groupby('relevance')['relevance'].count())



#----------------------------------------------------------------------------
# 1  BAr Plot of the relevance scores
#----------------------------------------------------------------------------

relv_array  = np.array(sorted(relv_frequency.items(), key=operator.itemgetter(0)))

plt.close('all')

fig, ax = plt.subplots(figsize=(10, 8))

ax.bar(range(relv_array.shape[0]),relv_array[:,1],align='center')

ax.set_xlabel('Relevance Scores')
ax.set_ylabel('Frequency')

fig.suptitle('Frequency of Relevance Scores(Response Variable)')

ax.set_xticklabels(relv_array[:,0])

ax.grid(True)

fig.savefig('relv_plot.png', bbox_inches='tight')

plt.show()
plt.close('all')


#----------------------------------------------------------------------------
# 2 Scatter plot of count of product_uid vs svg relevance
#----------------------------------------------------------------------------
pid_frequency = dict(all_data_df.groupby('product_uid')['product_uid'].count())
pid_avg_relv = train_df.groupby('product_uid', as_index=False)['relevance'].mean()
pid_avg_relv['pid_count'] = pid_avg_relv['product_uid'].map(lambda x: int(pid_frequency[x]))

#df = df[np.isfinite(df['EPS'])]

pid_avg_relv['subset'] = np.select([pid_avg_relv.relevance < 1.0, pid_avg_relv.relevance < 2.0, pid_avg_relv.relevance <= 3.0],
                         ['Relevance < 1', 'Relevance < 2', 'Relevance < 3'],'Others')

plt.close('all')

fig, ax = plt.subplots(figsize=(10, 8))

for color, label in zip('bgrm', ['Relevance < 1', 'Relevance < 2', 'Relevance < 3', 'Others']):
    subset = pid_avg_relv[pid_avg_relv.subset == label]
    ax.scatter(subset.relevance, subset.pid_count, s=120, c=color, label=str(label))

ax.legend()

ax.set_xlabel('Average Relevance Scores')
ax.set_ylabel('Product Id Frequency')

fig.suptitle('Average Relevance Scores by Frequency of Product ID')

ax.set_xticklabels(relv_array[:,0])

ax.grid(True)

fig.savefig('pid_relv_cnt_plot.png', bbox_inches='tight')

plt.show()
plt.close('all')


#----------------------------------------------------------------------------
# 3 Length of avg( jackard score ) Vs Relevance
#----------------------------------------------------------------------------

plt.close('all')

fig, ax = plt.subplots(figsize=(10, 8))

x = train_df['relevance']
for yft, clr, lwidth, lstyl, lbl in zip(
    ['jaccard_search_title',
     'jaccard_search_description',
     'jaccard_search_brand'] ,
    ['r', 'g', 'b', 'y'],
    [1, 2, 3, 4],
    ['-', '--', ':', '-.'],
    ['Jaccard - Search/Title',
     'Jaccard Search/Description',
     'Jaccard Search/Brand',
     'Words in Brand'] 
    ):
    y = train_df[yft]
    x_plot, y_plot = zip(*sorted((xVal, np.mean([yVal for a, yVal in zip(x,y) if xVal==a])) for xVal in set(x)))
    ax.plot(x_plot,y_plot,c=clr,label=lbl,linewidth=lwidth,linestyle=lstyl)

ax.legend()

ax.set_xlabel('Average Relevance Scores')

fig.suptitle('Jaccard scores for search in text by Relevance Score')

fig.savefig('jaccrd_relv_plot.png', bbox_inches='tight')

plt.show()
plt.close('all')


#----------------------------------------------------------------------------
# 4 count of words text features Vs Relevance
#----------------------------------------------------------------------------

plt.close('all')

fig, ax = plt.subplots(figsize=(10, 8))

x = train_df['relevance']
for yft, clr, lwidth, lstyl, lbl in zip(
    ['len_of_query',
     'len_of_description',
     'len_of_title'] ,
    ['r', 'g', 'b', 'y'],
    [1, 2, 3, 4],
    ['-', '--', ':', '-.'],
    ['Words in Search',
     'Words in Desc',
     'Words in Title',
     'Words in Brand'] 
    ):
    y = train_df[yft]
    x_plot, y_plot = zip(*sorted((xVal, np.mean([yVal for a, yVal in zip(x,y) if xVal==a])) for xVal in set(x)))
    ax.plot(x_plot,y_plot,c=clr,label=lbl,linewidth=lwidth,linestyle=lstyl)

ax.legend()

ax.set_xlabel('Average Relevance Scores')

fig.suptitle('Count of Words in text fields by Mean Relevance Scores')

fig.savefig('length_relv_plot.png', bbox_inches='tight')

plt.show()
plt.close('all')



#----------------------------------------------------------------------------
# 5 ratio matching words in search to total words in text feature Vs Relevance
#----------------------------------------------------------------------------

plt.close('all')

fig, ax = plt.subplots(figsize=(10, 8))

x = train_df['relevance']
for yft, clr, lwidth, lstyl, lbl in zip(
    ['ratio_brand',
     'ratio_description',
     'ratio_title',
     'ratio_brand_brand'] ,
    ['r', 'g', 'b', 'y'],
    [1, 2, 3, 4],
    ['-', '--', ':', '-.'],
    ['Ratio in Brand',
     'Ratio in Desc',
     'Ratio in Title',
     'Ratio of Brand in Brand'] 
    ):
    y = train_df[yft]
    x_plot, y_plot = zip(*sorted((xVal, np.mean([yVal for a, yVal in zip(x,y) if xVal==a])) for xVal in set(x)))
    ax.plot(x_plot,y_plot,c=clr,label=lbl,linewidth=lwidth,linestyle=lstyl)

ax.legend()

ax.set_xlabel('Average Relevance Scores')

fig.suptitle('Ratio of search words matching text by Mean Relevance Scores')

fig.savefig('ratio_relv_plot.png', bbox_inches='tight')

plt.show()
plt.close('all')








#----------------------------------------------------------------------------
# 6 matching word ngrams in search in text feature Vs Relevance
#----------------------------------------------------------------------------

plt.close('all')

fig, ax = plt.subplots(figsize=(10, 8))

x = train_df['relevance']
for yft, clr, lwidth, lstyl, lbl in zip(
    ['query_in_title',
     'query_in_description',
     'ratio_brand_brand'] ,
    ['r', 'g', 'b', 'y'],
    [1, 2, 3, 4],
    ['-', '--', ':', '-.'],
    ['Search NGrams in Title',
     'Search NGrams in Desc',
     'Ratio of Brand in Brand'] 
    ):
    y = train_df[yft]
    x_plot, y_plot = zip(*sorted((xVal, np.mean([yVal for a, yVal in zip(x,y) if xVal==a])) for xVal in set(x)))
    ax.plot(x_plot,y_plot,c=clr,label=lbl,linewidth=lwidth,linestyle=lstyl)

ax.legend()

ax.set_xlabel('Average Relevance Scores')

fig.suptitle('Count of matching search ngrams by MeanRelevance Scores')

fig.savefig('ngrams_relv_plot.png', bbox_inches='tight')

plt.show()
plt.close('all')





#----------------------------------------------------------------------------
# 7 seach last words in search in text feature Vs Relevance
#----------------------------------------------------------------------------

plt.close('all')

fig, ax = plt.subplots(figsize=(10, 8))

x = train_df['relevance']
for yft, clr, lwidth, lstyl, lbl in zip(
    ['query_last_word_in_title',
     'query_last_word_in_description'] ,
    ['r', 'g', 'b', 'y'],
    [1, 2, 3, 4],
    ['-', '--', ':', '-.'],
    ['Search Last Word in Title',
     'Search Last Word in Desc',
     'Ratio of Brand in Brand'] 
    ):
    y = train_df[yft]
    x_plot, y_plot = zip(*sorted((xVal, np.mean([yVal for a, yVal in zip(x,y) if xVal==a])) for xVal in set(x)))
    ax.plot(x_plot,y_plot,c=clr,label=lbl,linewidth=lwidth,linestyle=lstyl)

ax.legend()

ax.set_xlabel('Average Relevance Scores')

fig.suptitle('Count of text matching Last search words by Relevance Scores')

fig.savefig('lastw_relv_plot.png', bbox_inches='tight')

plt.show()
plt.close('all')






#----------------------------------------------------------------------------
# 8 seach segment in text feature Vs Relevance
#----------------------------------------------------------------------------

plt.close('all')

fig, ax = plt.subplots(figsize=(10, 8))

x = train_df['relevance']
for yft, clr, lwidth, lstyl, lbl in zip(
    ['search_term_feature'] ,
    ['r', 'g', 'b', 'y'],
    [1, 2, 3, 4],
    ['-', '--', ':', '-.'],
    ['Search Segments in Title',
     'Search Last Word in Title',
     'Search Last Word in Desc',
     'Ratio of Brand in Brand'] 
    ):
    y = train_df[yft]
    x_plot, y_plot = zip(*sorted((xVal, np.mean([yVal for a, yVal in zip(x,y) if xVal==a])) for xVal in set(x)))
    ax.plot(x_plot,y_plot,c=clr,label=lbl,linewidth=lwidth,linestyle=lstyl)

ax.legend()

ax.set_xlabel('Average Relevance Scores')

fig.suptitle('Count of text matching search segments by Relevance Scores')

fig.savefig('srch_seg_relv_plot.png', bbox_inches='tight')

plt.show()
plt.close('all')






exit()
