import nltk
import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from loadingReviews import loadReviews
from loadingReviews import textProcessing
from loadingReviews import generateTrainingTestSet

def bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=50):
    try:
        bigram_finder = BigramCollocationFinder.from_words(words)
        bigrams = bigram_finder.nbest(score_fn, n)
        return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])
    except ZeroDivisionError as detail:
        return dict([(word, True) for word in words])
        print 'Handling run-time error:', detail

def ngramNaiveBayesClassifier(reviews_training, reviews_test):    
    reviews_bigramtraining = []
    for r in reviews_training:
        reviews_bigramtraining.append((bigram_word_feats(r[0].split()), r[1]))
    
    reviews_bigramtest = []
    for r in reviews_test:
        reviews_bigramtest.append((bigram_word_feats(r[0].split()), r[1]))
    
    print("Training model....")
    classifier = nltk.NaiveBayesClassifier.train(reviews_bigramtraining)

    print("Running classifier....")
    test_labels = []
    prediction = []

    for r in reviews_bigramtest:
        test_labels.append(r[1])
        prediction.append(classifier.classify(r[0]))

    print("Results....")
    print(accuracy_score(test_labels, prediction))
    print(classification_report(test_labels, prediction))

#load reviews
reviews = loadReviews("imdbMovieReviews3.txt")

#bigram NV without title
processedReviews = textProcessing(reviews, False, 3)
result = generateTrainingTestSet(processedReviews, False)
reviews_training = result[0]
reviews_test = result[1]
print("NaiveBayes-Bigram without Review title....")
ngramNaiveBayesClassifier(reviews_training, reviews_test)

#bigram NV with title
processedReviews = textProcessing(reviews, False, 3)
result = generateTrainingTestSet(processedReviews, True)
reviews_training = result[0]
reviews_test = result[1]
print("NaiveBayes-Bigram with Review title....")
ngramNaiveBayesClassifier(reviews_training, reviews_test)

