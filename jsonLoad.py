import simplejson
from pprint import pprint

fd = open('imdbMovieReviews1.txt', 'r')
text = fd.read()
fd.close()
data = simplejson.loads(text)
	
pprint(data[0]["title"])