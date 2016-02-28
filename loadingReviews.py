import simplejson
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
    
def loadReviews(filename):
    print("Loading file....")
    fd = open(filename, 'r')
    text = fd.read()
    fd.close()
    data = simplejson.loads(text)

    #return all reviews as a list
    reviews = []    
    for i in data:
        reviews.extend(i["reviews"])    
    
    print("Total reviews:" + str(len(reviews)))
    return reviews

def textProcessing(reviews, removeStopWords, minLength):
    print("Text processing....")
    
    for review in reviews:
        if review["reviewContent"] != None:
            #select only words which include characters, no number, no punctuation
            tokenizer = RegexpTokenizer(r'[A-Za-z]+')
            #lowercase all words
            tokens = tokenizer.tokenize(review["reviewContent"].lower())
            
            filtered_words = tokens
            #remove stopwords
            if (removeStopWords):
                stopWordsSet = set(stopwords.words('english'))
                filtered_words = [w for w in tokens if not w in stopWordsSet]
            
            #remove words with less than "minLength" characters
            filtered_words = [e.lower() for e in filtered_words if len(e) >= minLength]
            review["reviewContent"] = " ".join(filtered_words)    
    return reviews
    
def generateTrainingTestSet(reviews, combineTitle):
    print("Create training & test data, 75%:training, 25%:test....")
    #get positive reviews
    pos_reviews = []
    for i in reviews:
        if i["reviewContent"] != None and int(i["reviewRating"]) >= 7 and int(i["reviewUseful"]) > 0:
            i["label"] = "positive"
            pos_reviews.append(i)

    #get negative reviews
    neg_reviews = []
    for i in reviews:
        if i["reviewContent"] != None and int(i["reviewRating"]) <= 4 and int(i["reviewRating"]) > 0 and int(i["reviewTotalVote"]) > 0:
            i["label"] = "negative"
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
        p["label"] = "positive"

    #get negative reviews training
    #neg_reviews_training = sorted(neg_reviews, key=lambda k: int(k['reviewTotalVote']), reverse=True)
    #neg_reviews_training = neg_reviews_training[:2500]
    #neg_reviews_training = sorted(neg_reviews_training, key=lambda k: int(k['reviewUseful'])/int(k['reviewTotalVote']), reverse=True)
    neg_reviews = sorted(neg_reviews, key=lambda k: int(k['reviewUseful']), reverse=True)
    neg_reviews_training = neg_reviews[:(len(neg_reviews) * 3/4)]
    neg_reviews_test = neg_reviews[(len(neg_reviews) * 3/4):]
    for n in neg_reviews_training:
        n["label"] = "negative"

    print("No pos reviews training: " + str(len(pos_reviews_training)))
    print("No pos reviews test: " + str(len(pos_reviews_test)))
    print("No neg reviews training: " + str(len(neg_reviews_training)))
    print("No neg reviews test: " + str(len(neg_reviews_test)))

    reviews_training = []
    for r in pos_reviews_training + neg_reviews_training:
        if(combineTitle and r["reviewTitle"] != None):
            r["reviewContent"] = r["reviewContent"] + " " + r["reviewTitle"]
        reviews_training.append((r["reviewContent"], r["label"]))
    
    reviews_test = []
    for r in pos_reviews_test + neg_reviews_test:
        if(combineTitle and r["reviewTitle"] != None):
            r["reviewContent"] = r["reviewContent"] + " " + r["reviewTitle"]
        reviews_test.append((r["reviewContent"], r["label"]))    
    return reviews_training, reviews_test