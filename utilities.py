from __future__ import division

import pandas as pd
import numpy as np
import re
import itertools
import operator

from pyspark.ml.classification import LogisticRegression
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint

def readCsv(fileName, topic=None):
    df = pd.read_csv(fileName)
    df = df[df.tweet_text != "Not Available"]
    df = df[df.classification != 'neutral']

    return df

def clean_tweet(tweet):
    '''
    Utility function to clean the text in a tweet by removing 
    links and special characters using regex.
    '''
    return ' '.join(re.sub("(@[A-Za-z]+)|([^A-Za-z \t])|(\\w+:\\/\\/\\S+)", " ", tweet).split()).lower()

def split_keys(df_keys):
    splitted = []

    for string in df_keys['key_phrases']:
        words = []
        s = str(string)
        t = s.replace(',', ' ')
        s = t.split()

        for word in s:
            words.append(word)
        splitted.append(words)

    return splitted

def comb_list(df_set, df_keys):
    comb = []

    for i in range(0, df_set.shape[0]):
        l_comb = []

        for j in range(1, len(df_keys[i]) + 1):
            l_comb.append(list(itertools.combinations(df_keys[i], j)))
        comb.append(l_comb)

    return comb

def combine_tweets(df_set, list_combinations):
    tweet_combinations = []

    for i in range(0, df_set.shape[0]):
        comb_t = []        

        for l in list_combinations[i]:            
            for word in l:
                comb_t.append(clean_tweet(str(df_set.tweet_text[i]) + " " + str(word)))

        tweet_combinations.append(comb_t)
    
    return tweet_combinations

def getSimilar(df_train, df_test, topicToCheck, test_size=0.2):
    cat_count = {}
    topic_cat = set()
    topic_class_tuples = []
    final_list = []
    
    topic_train = df_train[df_train.topic == topicToCheck]
    
    # Controllo quanti esempi ci sono per classe nell'insieme di train e inizializzo i dizionari di appoggio
    for top_c in np.unique(topic_train['class']):
        topic_class_tuples.append((top_c, topic_train[topic_train['class'] == top_c].shape[0]))
        dd = {}
        cat_count[top_c] = dd    
    
    # Mi salvo tutte le categorie del topic
    for row in topic_train.categories:
        for word in row.split():
            topic_cat.add(word)
    
    # Numero di sample da restituire nel test set
    n_samples = int((df_train[df_train.topic == topicToCheck].shape[0])*test_size/(1-test_size))


    for index, row in df_test.iterrows():
        cl = row['class']
        count = 0

        try:        
            for word in row['categories'].split():                  
                if word in topic_cat:
                    count = count + 1
                    
            cat_count[cl][index] = count
        
        except IndexError:
            count = 0

        except KeyError:
            count = 0
    
    for k in cat_count.keys():

        l_cl = sorted(cat_count[k].items(), key=operator.itemgetter(1), reverse=True)

        # print(l_cl)

        # [x[1] for x in pqrst if x[0] == 0.0][0]

        # perc_cl = topic_class_tuples[int(k)][1]/topic_train.shape[0]
        perc_cl = (([x[1] for x in topic_class_tuples if x[0] == k][0]))/(topic_train.shape[0])
        n_sampl_cl = int(perc_cl*n_samples)
        #print(n_sampl_cl)

        [final_list.append(x[0]) for x in l_cl[0:n_sampl_cl]]

    return (final_list, n_samples)

def getConfusionMatrix(pred):

    scores = pred.select('class', 'prediction').toPandas()

    tp = (scores['class'] == 1) & (scores['prediction'] == 1)
    tn = (scores['class'] == 0) & (scores['prediction'] == 0)
    fp = (scores['class'] == 0) & (scores['prediction'] == 1)
    fn = (scores['class'] == 1) & (scores['prediction'] == 0)

    return (scores.loc[tp].shape[0], scores.loc[tn].shape[0], scores.loc[fp].shape[0], scores.loc[fn].shape[0])

def task_B(df_train, df_test, sqlContext, pipe, features_col):    
    avg_rec = []
    f1_score = []
    accuracy = []

    i = 0
    test_size = 0.4

    lr = LogisticRegression().setLabelCol('class').setFeaturesCol(features_col).setRegParam(0.0).setMaxIter(100).setElasticNetParam(0.)

    print("Numero di topic da processare:", len(np.unique(df_train.topic)))

    for topic in np.unique(df_train.topic):
        index_test = getSimilar(df_train, df_test, topic, test_size)[0]   

        topic_train = df_train[df_train.topic == topic]
        topic_test = df_test.iloc[index_test]

        # print(topic_train.shape, topic_test.shape)

        df_f = sqlContext.createDataFrame(pd.concat([topic_train, topic_test]))

        tr3_test = pipe.fit(df_f).transform(df_f)

        train_topic = tr3_test.filter(tr3_test['topic'] == topic)
        test_topic = tr3_test.filter(tr3_test['topic'] != topic) 

        pred = lr.fit(train_topic).transform(test_topic)
        # pred.show()

        tp, tn, fp, fn = getConfusionMatrix(pred)

        try:
            p_pos = tp / (tp + fp)
        except ZeroDivisionError:
            p_pos = 0

        try:
            p_neg = tn / (tn + fn)
        except ZeroDivisionError:
            p_neg = 0

        acc_aux = (tp + tn) / (tp + tn + fp + fn)
        f1_aux = (2*tp) / ((2*tp) + fp + fn)
        avg_rec_aux = ( p_pos + p_neg ) / 2

        accuracy.append(acc_aux)
        f1_score.append(f1_aux)
        avg_rec.append(avg_rec_aux)        

        i = i + 1

        if i % 50 == 0:
            print("Topic processati:", i)              
        
    return (np.mean(accuracy), np.mean(f1_score), np.mean(avg_rec))

def mae_ms(scores):
    n = 5
    # scores = pred.select('class', 'prediction').toPandas()    
    aux_sum = 0;
    mae_first = 0;

    hp_c = scores[scores['class'] == 4].shape[0]
    p_c = scores[scores['class'] == 3].shape[0]
    ok_c = scores[scores['class'] == 2].shape[0]
    n_c = scores[scores['class'] == 1].shape[0]
    hn_c = scores[scores['class'] == 0].shape[0]
    
    try:
        mae_first = mae_first + (1/hp_c)
    except ZeroDivisionError:
        mae_first = mae_first + 0
     
    try:
        mae_first = mae_first + (1/p_c)
    except ZeroDivisionError:
        mae_first = mae_first + 0
        
    try:
        mae_first = mae_first + (1/ok_c)
    except ZeroDivisionError:
        mae_first = mae_first + 0
        
    try:
        mae_first = mae_first + (1/n_c)
    except ZeroDivisionError:
        mae_first = mae_first + 0
        
    try:
        mae_first = mae_first + (1/hn_c)
    except ZeroDivisionError:
        mae_first = mae_first + 0
        
    for index, rows in scores.iterrows():
        aux_sum = aux_sum + abs(rows['prediction'] - rows['class'])
        
    maem = (1/n) * mae_first * aux_sum
    maeni = (1/scores.shape[0]) * aux_sum
    
    return (maem, maeni)

def task_C(df_train, df_test, sc, sqlContext, pipe, features_col):
    maem = []
    maeni = []
    i = 0
    
    for topic in np.unique(df_train.topic):    
        pred = pd.DataFrame(columns=['class', 'prediction'])
        dtc = DecisionTree()
        sparse_data = []
        test_data = []
        cl_cl = []
            
        index_test = getSimilar(df_train, df_test, topic, test_size=0.3)[0]
        
        topic_train = df_train[df_train.topic == topic]
        topic_test = df_test.iloc[index_test]
        
        topic_train.loc[:, 'class'] = topic_train['class'].map(lambda x : x+2)
        topic_test.loc[:, 'class'] = topic_test['class'].map(lambda x : x+2)
        
        df_f = sqlContext.createDataFrame(pd.concat([topic_train, topic_test]))

        tr3_test = pipe.fit(df_f).transform(df_f)

        train_topic = tr3_test.filter(tr3_test['topic'] == topic)
        test_topic = tr3_test.filter(tr3_test['topic'] != topic)    

        
        # train_topic = MLUtils.convertVectorColumnsFromML(train_topic, features_col)
        # test_topic = MLUtils.convertVectorColumnsFromML(test_topic, features_col)

        for index, row in train_topic.toPandas().iterrows():
            sparse_data.append(LabeledPoint(float(row['class']), row[features_col]))

        for index, row in test_topic.toPandas().iterrows():
            cl_cl.append(row['class'])
            test_data.append(row[features_col])

        model = DecisionTree.trainClassifier(sc.parallelize(sparse_data), 5, {})

        pred['class'] = cl_cl
        pred['prediction'] = model.predict(sc.parallelize(test_data)).collect()
        
        maem_aux, maeni_aux = mae_ms(pred)
        
        maem.append(maem_aux)
        maeni.append(maeni_aux)
        
        i = i + 1

        if i % 25 == 0:
            print("Topic processati:", i)        
        
    return (np.mean(maem), np.mean(maeni))