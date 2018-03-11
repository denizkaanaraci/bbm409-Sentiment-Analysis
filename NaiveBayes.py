import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import math
import time


# parses tweets according to their classes(neg, neut, pos)
def parseTweets(tweetList):
    negativeList, neutralList, positiveList = [], [], []
    for i in range(len(tweetList)):
        if tweetList[i][0] == b'0':
            negativeList.append(BoWVectorList[i])
        elif tweetList[i][0] == b'2':
            neutralList.append(BoWVectorList[i])
        elif tweetList[i][0] == b'4':
            positiveList.append(BoWVectorList[i])

    return np.array(negativeList), np.array(neutralList), np.array(positiveList)


# vectorize the tweet with delimiters and BoW options(ngram_range (1,1=unigram, 2,2=bigram, 1,2 = both))
def TweetVectorizer(tweetList):
    # min_df and max_df changes parse options of word frequency
    # ngram_range changes parse options of n-gram
    vector = CountVectorizer(min_df=4, max_df=150, ngram_range=(1, 1))
    vectorList = vector.fit_transform(tweetList[:, 1])

    return vector, vectorList.toarray()


# creates dictionary with feature names(words) and word counts
def DictMaker(vectorList):
    VectDict = dict(zip(BoWVector.get_feature_names(), np.asarray(vectorList.sum(axis=0)).ravel()))

    return VectDict


# conditional probability calculation
def condProb(value, Dict):
    Len = len(BoWDict)
    Sum = sum(Dict.values())

    try:
        return (value + 1) / (Sum + Len)  # if dictionary has key
    except:
        return (0 + 1) / (Sum + Len)  # if dictionary has not key


# calculates naive bayes
def nBayes(tweetDict, negativeBoWDict, neutralBoWDict, positiveBoWDict):
    negProb = math.log10(len(negativeVectorList) / len(trainTweets))
    neutProb = math.log10(len(neutralVectorList) / len(trainTweets))
    posProb = math.log10(len(positiveVectorList) / len(trainTweets))

    for word in tweetDict:
        try:
            val = negativeBoWDict[word]
            negProb += math.log10(np.power(condProb(val, negativeBoWDict), val))
        except:
            negProb += math.log10(condProb(0, negativeBoWDict))
        try:
            val = neutralBoWDict[word]
            neutProb += math.log10(np.power(condProb(val, neutralBoWDict), val))
        except:
            neutProb += math.log10(condProb(0, neutralBoWDict))
        try:
            val = positiveBoWDict[word]
            posProb += math.log10(np.power(condProb(val, positiveBoWDict), val))
        except:
            posProb += math.log10(condProb(0, positiveBoWDict))

    return negProb, neutProb, posProb


# calculates prediction
def Prediction(negativeBoWDict, neutralBoWDict, positiveBoWDict):
    correctPredict = 0

    for i in range(len(validationTweets)):

        vectorizer = CountVectorizer(ngram_range=(1, 1))
        # this try except block catches empty tweet vectors.(one word tweets or only stop word tweets)
        try:
            twitVector = vectorizer.fit_transform(validationTweets[i:i + 1, 1])
        except:
            continue

        tweetDict = dict(zip(vectorizer.get_feature_names(), np.asarray(twitVector.sum(axis=0)).ravel()))

        negProb, neutProb, posProb = nBayes(tweetDict, negativeBoWDict, neutralBoWDict, positiveBoWDict)

        if min(negProb, neutProb, posProb) == negProb and validationTweets[i][0] == b'0':
            correctPredict += 1
        elif min(negProb, neutProb, posProb) == neutProb and validationTweets[i][0] == b'2':
            correctPredict += 1
        elif min(negProb, neutProb, posProb) == posProb and validationTweets[i][0] == b'4':
            correctPredict += 1

    percentage = (correctPredict / len(validationTweets)) * 100

    return percentage


def main():
    # makes negative, neutral and positive Bag Of Words
    negativeBoWDict = DictMaker(negativeVectorList)
    neutralBoWDict = DictMaker(neutralVectorList)
    positiveBoWDict = DictMaker(positiveVectorList)

    # calculating the accuracy
    Accuracy = Prediction(negativeBoWDict, neutralBoWDict, positiveBoWDict)

    print("Accuracy is " + str(Accuracy) + "%")


trainTweets = np.load("train_tweets.npy")
validationTweets = np.load("validation_tweets.npy")

BoWVector, BoWVectorList = TweetVectorizer(trainTweets)
BoWDict = DictMaker(BoWVectorList)  # creates Bag of Word dictionary with training data

# train tweets splits according to their classes
negativeVectorList, neutralVectorList, positiveVectorList = parseTweets(trainTweets)

main()
