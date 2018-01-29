import pandas as pd
import numpy as np
import re
import itertools
import operator

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

def getSimilar(df_train, df_test, topicToCheck):
    cat_count = {}
    topic_cat = set()
    
    for row in df_train[df_train.topic == topicToCheck].categories:
        for word in row.split():
            topic_cat.add(word)
    
    n_samples = int((df_train[df_train.topic == topicToCheck].shape[0])*20/80)

#    print(topicToCheck, "ha", str(df_train[df_train.topic == topicToCheck].shape[0]), "elementi, quindi si farà il test per", n_samples, "elementi")

    for index, row in df_test.iterrows():
        cat_count[index] = 0;
        for word in row['categories'].split():        
            if word in topic_cat:
                cat_count[index] = cat_count[index] + 1


    sorted_x = sorted(cat_count.items(), key=operator.itemgetter(1), reverse=True)
    
    final_test_sorted = [x[0] for x in sorted_x[0:n_samples]]

#    print("Done!")
    
    return (final_test_sorted, n_samples)

def getConfusionMatrix(pred):

    scores = pred.select('class', 'prediction').toPandas()
    tp = (scores['class'] == 1) & (scores['prediction'] == 1)
    tn = (scores['class'] == 0) & (scores['prediction'] == 0)
    fp = (scores['class'] == 0) & (scores['prediction'] == 1)
    fn = (scores['class'] == 1) & (scores['prediction'] == 0)

    return (scores.loc[tp].shape[0], scores.loc[tn].shape[0], scores.loc[fp].shape[0], scores.loc[fn].shape[0])
    