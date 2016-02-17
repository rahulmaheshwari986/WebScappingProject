import simplejson
import nltk
import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
import collections

def bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=100):
	bigram_finder = BigramCollocationFinder.from_words(words)
	bigrams = bigram_finder.nbest(score_fn, n)
	return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])
	
print("Loading file....")
fd = open('imdbMovieReviews3.txt', 'r')
text = fd.read()
fd.close()
data = simplejson.loads(text)

reviews = []
print("Reading reviews....")
for i in data:
	reviews.extend(i["reviews"])

for review in reviews:
#	if review["reviewContent"] != None:
#		tokenizer = RegexpTokenizer(r'\w+')
#		tokens = tokenizer.tokenize(review["reviewContent"].lower())
		#filtered_words = [w for w in tokens if not w in set(stopwords.words('english'))]
		review["reviewContent"] = review["reviewContent"].lower()

print("Separating Pos Neg reviews....")
#get positive reviews
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

print("No pos reviews: " + str(len(pos_reviews)))
print("No neg reviews: " + str(len(neg_reviews)))

#get positive reviews training
#pos_reviews_training = sorted(pos_reviews, key=lambda k: int(k['reviewTotalVote']), reverse=True)
#pos_reviews_training = pos_reviews_training[:2500]
#pos_reviews_training = sorted(pos_reviews_training, key=lambda k: int(k['reviewUseful'])/int(k['reviewTotalVote']), reverse=True)
pos_reviews = sorted(pos_reviews, key=lambda k: int(k['reviewUseful']), reverse=True)
pos_reviews_training = pos_reviews[:(len(pos_reviews) * 3/4)]
pos_reviews_test = pos_reviews[(len(pos_reviews) * 3/4):]
for p in pos_reviews_training:
	p["sentiment"] = "positive"

#get negative reviews training
#neg_reviews_training = sorted(neg_reviews, key=lambda k: int(k['reviewTotalVote']), reverse=True)
#neg_reviews_training = neg_reviews_training[:2500]
#neg_reviews_training = sorted(neg_reviews_training, key=lambda k: int(k['reviewUseful'])/int(k['reviewTotalVote']), reverse=True)
neg_reviews = sorted(neg_reviews, key=lambda k: int(k['reviewUseful']), reverse=True)
neg_reviews_training = neg_reviews[:(len(neg_reviews) * 3/4)]
neg_reviews_test = neg_reviews[(len(neg_reviews) * 3/4):]
for n in neg_reviews_training:
	n["sentiment"] = "negative"

print("No pos reviews training: " + str(len(pos_reviews_training)))
print("No pos reviews test: " + str(len(pos_reviews_test)))
print("No neg reviews training: " + str(len(neg_reviews_training)))
print("No neg reviews test: " + str(len(neg_reviews_test)))

reviews_training = []
for r in pos_reviews_training + neg_reviews_training:
	#words_filtered = [e.lower() for e in r["reviewContent"].split() if len(e) >= 3]
	reviews_training.append((bigram_word_feats(r["reviewContent"].split()), r["sentiment"]))
	

for r in pos_reviews_test + neg_reviews_test:
	#words_filtered = [e.lower() for e in r["reviewContent"].split() if len(e) >= 3]
	words_filtered = [e.lower() for e in r["reviewContent"].split() if len(e) >= 3]
	r["reviewContent"] = bigram_word_feats(r["reviewContent"].split())

print("Training model....")
#reviews_training_set = nltk.classify.apply_features(extract_features, reviews_training)
classifier = nltk.NaiveBayesClassifier.train(reviews_training)

print("Running classifier....")
result = []
refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)

for r, (r["reviewContent"], r["userSentiment"]) in pos_reviews_test + neg_reviews_test:
	refsets[r["userSentiment"]].add(r)
	#r["naiveBayesClassifierResult"] = classifier.classify(extract_features(r["reviewContent"].split()))
	r["naiveBayesClassifierResult"] = classifier.classify(r["reviewContent"])
	observed = r["naiveBayesClassifierResult"]
	testsets[observed].add(r["reviewContent"])
	result.append(r)


print("Results....")

print 'accuracy:', nltk.classify.util.accuracy(classifier, testsets)
print 'pos precision:', nltk.metrics.precision(refsets['positive'], testsets['positive'])
print 'pos recall:', nltk.metrics.recall(refsets['positive'], testsets['positive'])
print 'neg precision:', nltk.metrics.precision(refsets['negative'], testsets['negative'])
print 'neg recall:', nltk.metrics.recall(refsets['negative'], testsets['negative'])

json_data = simplejson.dumps(result, indent=4, skipkeys=True, sort_keys=True, default=lambda o: o.__dict__)
fd = open('result-content.txt', 'w')
fd.write(json_data)
fd.close()
