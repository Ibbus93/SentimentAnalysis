def deleteNotAvailable(tweets):
    av = []
    for tweet in tweets:
        if tweet[3] != 'Not Available' and tweet[2] != 'off topic':
            av.append(tweet)
    return av

def loadTsv(file, sc):
    aux = deleteNotAvailable(
        sc.textFile(file) \
            .map(lambda x: x.encode("ascii", "ignore")) \
            .map(lambda line: line.split("\t")) \
            .collect()
    )

    # Trasformo l'attributo categorico
    score = {'negative': -1, 'neutral': 0, 'positive': 1}
    for x in aux:
        x[2] = score[x[2]]

    return sc.parallelize(aux)
    # return aux

# t2015test = utilities.loadTsv("tweets_subtask_BD/twitter-2015test-BD-down.tsv", sc)
# t2016dev = utilities.loadTsv("tweets_subtask_BD/twitter-2016dev-BD-down.tsv", sc)
# t2016devtest = utilities.loadTsv("tweets_subtask_BD/twitter-2016devtest-BD-down.tsv", sc)
# t2016test = utilities.loadTsv("tweets_subtask_BD/twitter-2016test-BD-down.tsv", sc)
# t2016train = utilities.loadTsv("tweets_subtask_BD/twitter-2016train-BD-down.tsv", sc)