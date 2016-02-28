import nltk
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from loadingReviews import loadReviews
from loadingReviews import textProcessing
from loadingReviews import generateTrainingTestSet

def unigramNaiveBayesClassifier(reviews_training, reviews_test):
	reviews_trainingsplit = []
	for r in reviews_training:
		reviews_trainingsplit.append((r[0].split(), r[1]))
	
	all_words = []
	for r in reviews_trainingsplit:
		all_words.extend(r[0])

	wordlist = nltk.FreqDist(all_words)
	word_features = set(wordlist.keys())
	
	#Function to extract unigram
	def extract_features(document):
		document_words = set(document)
		features = {}
		for word in word_features:
			features['contains(%s)' % word] = (word in document_words)
		return features
	
	print("Training model....")
	reviews_training_set = nltk.classify.apply_features(extract_features, reviews_trainingsplit)
	classifier = nltk.NaiveBayesClassifier.train(reviews_training_set)

	print("Running classifier....")
	test_labels = []
	prediction = []
	
	for r in reviews_test:
		test_labels.append(r[1])	
		prediction.append(classifier.classify(extract_features(r[0].split())))

	print("Results....")
	print(accuracy_score(test_labels, prediction))
	print(classification_report(test_labels, prediction))
	
#load reviews
reviews = loadReviews("imdbMovieReviews3.txt")

#unigram NV without title
processedReviews = textProcessing(reviews, False, 3)
result = generateTrainingTestSet(processedReviews, False)
reviews_training = result[0]
reviews_test = result[1]
print("NaiveBayes-Unigram without Review title....")
unigramNaiveBayesClassifier(reviews_training, reviews_test)

#unigram NV without title and remove stopwords
processedReviews = textProcessing(reviews, True, 3)
result = generateTrainingTestSet(processedReviews, False)
reviews_training = result[0]
reviews_test = result[1]
print("NaiveBayes-Unigram without Review title and without stopwords....")
unigramNaiveBayesClassifier(reviews_training, reviews_test)

#unigram NV with title and stopwords
processedReviews = textProcessing(reviews, False, 3)
result = generateTrainingTestSet(processedReviews, True)
reviews_training = result[0]
reviews_test = result[1]
print("NaiveBayes-Unigram with Review title and with stopwords....")
unigramNaiveBayesClassifier(reviews_training, reviews_test)

