# You need to install scikit-learn:
# sudo pip install scikit-learn
#
# Dataset: Polarity dataset v2.0
# http://www.cs.cornell.edu/people/pabo/movie-review-data/
#
# Full discussion:
# https://marcobonzanini.wordpress.com/2015/01/19/sentiment-analysis-with-python-and-scikit-learn

import time
import simplejson

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report

print("Loading file....")
fd = open('imdbMovieReviews3.txt', 'r')
text = fd.read()
fd.close()
data = simplejson.loads(text)

reviews = []
for i in data:
    reviews.extend(i["reviews"])

pos_reviews = []
for i in reviews:
    if i["reviewContent"] != None and int(i["reviewRating"]) >= 7 and int(i["reviewUseful"]) > 0:
        i["userSentiment"] = "positive"
        pos_reviews.append(i)

#get negative reviews
neg_reviews = []
for i in reviews:
    if i["reviewContent"] != None and int(i["reviewRating"]) <= 4 and int(i["reviewRating"]) > 0 and int(i["reviewTotalVote"]) > 0:
        i["userSentiment"] = "negative"
        neg_reviews.append(i)

pos_reviews = sorted(pos_reviews, key=lambda k: int(k['reviewUseful']), reverse=True)
pos_reviews_training = pos_reviews[:(len(pos_reviews) * 3/4)]
pos_reviews_test = pos_reviews[(len(pos_reviews) * 3/4):]

neg_reviews = sorted(neg_reviews, key=lambda k: int(k['reviewUseful']), reverse=True)
neg_reviews_training = neg_reviews[:(len(neg_reviews) * 3/4)]
neg_reviews_test = neg_reviews[(len(neg_reviews) * 3/4):]


train_data = []
train_labels = []
test_data = []
test_labels = []

for r in pos_reviews_training:
    train_data.append(r["reviewContent"])
    train_labels.append("positive")

for r in neg_reviews_training:
    train_data.append(r["reviewContent"])
    train_labels.append("negative")

for r in pos_reviews_test:
    test_data.append(r["reviewContent"])
    test_labels.append("positive")

for r in neg_reviews_test:
    test_data.append(r["reviewContent"])
    test_labels.append("negative")


    # Create feature vectors
print("Create feature vectors...")

vectorizer = TfidfVectorizer(min_df=5, max_df = 0.8, sublinear_tf=True, use_idf=True, decode_error='ignore', strip_accents='unicode', ngram_range=(1, 3))
train_vectors = vectorizer.fit_transform(train_data)
test_vectors = vectorizer.transform(test_data)


    # Perform classification with SVM, kernel=linear
print("Perform classification with SVM.LinearSVC")

classifier_liblinear = svm.LinearSVC()
t0 = time.time()
classifier_liblinear.fit(train_vectors, train_labels)
t1 = time.time()
prediction_liblinear = classifier_liblinear.predict(test_vectors)
t2 = time.time()
time_liblinear_train = t1-t0
time_liblinear_predict = t2-t1


    # Print results in a nice table
print("Results for LinearSVC()")
print("Training time: %fs; Prediction time: %fs" % (time_liblinear_train, time_liblinear_predict))
print(classification_report(test_labels, prediction_liblinear))