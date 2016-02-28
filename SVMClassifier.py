import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from nltk.tokenize.regexp import wordpunct_tokenize
#from nltk.stem.wordnet import WordNetLemmatizer
#from nltk.tokenize import word_tokenize
#from nltk import pos_tag
#from nltk.corpus import wordnet as wn
from loadingReviews import loadReviews
from loadingReviews import generateTrainingTestSet

def tokenize(x):
    return [w for w in wordpunct_tokenize(x) if len(w)>=3]

#def get_wordnet_pos(pos):
#    wordnet_tag = {'NN':wn.NOUN, 'JJ':wn.ADJ, 'VB':wn.VERB, 'RB':wn.ADV}
#    try:
#        return wordnet_tag[pos[:2]]
#    except:
#        return None
        
#class LemmaTokenizer(object):
#    def __init__(self):
#        self.wnl = WordNetLemmatizer()
        
#    def __call__(self, doc):
#        lemmas = []
#        for word, pos in pos_tag(word_tokenize(doc)):
#            if len(word) >= 3:
#                pos = get_wordnet_pos(pos)
#                if (pos != None):
#                    lemmas.append(self.wnl.lemmatize(word, pos))
#                else:
#                    lemmas.append(word)
#        return lemmas
    
def linearSVMClassifier(reviews_training, reviews_test):
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    for r in reviews_training:
        train_data.append(r[0])
        train_labels.append(r[1])

    for r in reviews_test:
        test_data.append(r[0])
        test_labels.append(r[1])

    # Create feature vectors
    print("Create feature vectors...")

    vectorizer = TfidfVectorizer(min_df=3, max_df = 0.8, sublinear_tf=True, use_idf=True, 
                             decode_error='ignore', strip_accents='unicode', ngram_range=(1, 2), tokenizer=tokenize)
    train_vectors = vectorizer.fit_transform(train_data)
    test_vectors = vectorizer.transform(test_data)


    # Perform classification with SVM, Linear SVC
    print("Perform classification with SVM.LinearSVC")

    classifier_liblinear = svm.LinearSVC()
    t0 = time.time()
    classifier_liblinear.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_liblinear = classifier_liblinear.predict(test_vectors)
    t2 = time.time()
    time_liblinear_train = t1-t0
    time_liblinear_predict = t2-t1


    # Print results in tabular format
    print("Results for LinearSVC()")
    print("Training time: %fs; Prediction time: %fs" % (time_liblinear_train, time_liblinear_predict))
    print(accuracy_score(test_labels, prediction_liblinear))
    print(classification_report(test_labels, prediction_liblinear))

#load reviews
reviews = loadReviews("imdbMovieReviews3.txt")

#SVM without title
result = generateTrainingTestSet(reviews, False)
reviews_training = result[0]
reviews_test = result[1]
print("SVM without Review title....")
linearSVMClassifier(reviews_training, reviews_test)
    
#SVM with title
result = generateTrainingTestSet(reviews, True)
reviews_training = result[0]
reviews_test = result[1]
print("SVM with Review title....")
linearSVMClassifier(reviews_training, reviews_test)