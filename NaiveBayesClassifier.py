import string
import simplejson
from pprint import pprint
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

#fd = open('imdbMovieReviews1.txt', 'r')
#text = fd.read()
#fd.close()
#data = simplejson.loads(text)
	

#for i in data:
#	reviews.extend(i["reviews"])
	
print("Loading file....")
fd = open('imdbMovieReviews3.txt', 'r')
text = fd.read()
fd.close()
data = simplejson.loads(text)

reviews = []
print("Remove punctuation, stop words....")
for i in data:
	reviews.extend(i["reviews"])

for review in reviews:
	tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(review["reviewContent"])
	filtered_words = [w for w in tokens if not w in set(stopwords.words('english'))]
	review["reviewContent"] = " ".join(filtered_words)

print("Separate Pos Neg reviews....")
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
	words_filtered = [e.lower() for e in r["reviewContent"].split() if len(e) >= 3]
	reviews_training.append((words_filtered, r["sentiment"]))

all_words = []
for r in reviews_training:
	all_words.extend(r[0])

wordlist = nltk.FreqDist(all_words)
word_features = wordlist.keys()

def extract_features(document):
	document_words = set(document)
	features = {}
	for word in word_features:
		features['contains(%s)' % word] = (word in document_words)
	return features
 
def bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])

print("Training model....")
reviews_training_set = nltk.classify.apply_features(bigram_word_feats, reviews_training)
print("Running classifier....")
classifier = nltk.NaiveBayesClassifier.train(reviews_training_set)

print("Results....")
result = []
for r in pos_reviews_test + neg_reviews_test:
	r["naiveBayesClassifierResult"] = classifier.classify(extract_features(r["reviewContent"].split()))
	result.append(r)

tp = 0
tn = 0
fp = 0
fn = 0
for res in result:
	if res["userSentiment"] == res["naiveBayesClassifierResult"]:
		if res["userSentiment"] == "positive":
			tp = tp + 1
		else:
			tn = tn + 1
	else:
		if res["userSentiment"] == "positive":
			fn = fn + 1
		else:
			fp = fp + 1

print("TP : " + str(tp))
print("TN : " + str(tn))
print("FP : " + str(fp))
print("FN : " + str(fn))
print("Accuracy : ")
print((tp+tn)/(tp+tn+fp+fn))

fd = open('result-title.txt', 'w')
fd.write(result)
fd.close()





