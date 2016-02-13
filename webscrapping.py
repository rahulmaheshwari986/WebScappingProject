import urllib2
from bs4 import BeautifulSoup
import json
import simplejson

class Movie:
	title = ""
	rating = 0
	datePublished = ""
	year = ""
	directors = []
	actors = []
	reviews = []
	
	def __init__(self, title, rating, datePublished, year, directors, actors, reviews):
		self.title = title
		self.rating = rating
		self.datePublished = datePublished
		self.year = year
		self.directors = directors
		self.actors = actors
		self.reviews = reviews
		
class Review:
	reviewTitle = ""
	reviewAuthor = ""
	reviewDate = ""
	reviewRating = -1
	reviewUseful = 0
	reviewTotalVote = 0
	reviewContent = ""
	
	def __init__(self, reviewTitle, reviewAuthor, reviewDate, reviewRating, reviewUseful, reviewTotalVote, reviewContent):
		self.reviewTitle = reviewTitle
		self.reviewAuthor = reviewAuthor
		self.reviewDate = reviewDate
		self.reviewRating = reviewRating
		self.reviewUseful = reviewUseful
		self.reviewTotalVote = reviewTotalVote
		self.reviewContent = reviewContent

def cleanText(text):
    text = text.replace("<br>", "")
    text = text.replace("\n", " ")
    text = text.replace("\r", "")
    return text

def getMovieTitlesByGenres(genres, noMovies):
	step = 50
	count = 0
	all_movie_ids = []
	
	while count < noMovies:
		imdbLink = "http://www.imdb.com/search/title?genres=" + genres + "&sort=moviemeter,asc&start=" + str(count + 1) + "&title_type=feature"
		page = urllib2.urlopen(imdbLink)
		soup = BeautifulSoup(page)

		all_spans = soup.findAll("span", { "class" : "wlb_wrapper" })

		for i in all_spans:
			all_movie_ids.append(i.get("data-tconst"))
	
		count = count + step	
	return all_movie_ids

def getMovieReviews(link):
    Review_List = []

    for ctr in range(0, 5):
        numlink = link + str(ctr*10)
        review_soup = BeautifulSoup(urllib2.urlopen(numlink))
        text_tags = review_soup.find_all("p", attrs={'class' : None})
        
        Reviews = review_soup.find_all("div", attrs={'class' : None, 'id' : None, 'style' : None})
        reviewTitleList = []
        reviewAuthorList = []
        reviewDateList = []
        reviewRatingList = []
        reviewUsefulList = []
        reviewTotalVoteList = []
        reviewList = []
    
        for review in Reviews:	
            small_tags = review.find_all("small")
            anchor_tags = review.find_all("a")
            header_tags = review.find_all("h2")
            img_tags = review.find_all("img")
    
            if (len(small_tags) != 0):
                reviewTitleList.append(header_tags[0].find(text=True)) #Title
                reviewAuthorList.append(anchor_tags[1].find(text=True)) #Author

                if(len(small_tags) == 3):                          #Date 
                    reviewDateList.append(small_tags[2].find(text=True))
                elif len(small_tags) == 2:
                    reviewDateList.append(small_tags[1].find(text=True))
                else:
                	reviewDateList.append(small_tags[0].find(text=True))
                
                #format of useful reviews "7 out of 11 people found the following review useful:"	
                if "found the following review useful" in small_tags[0].find(text=True):
                	words = small_tags[0].find(text=True).split()
                	reviewUsefulList.append(words[0])  #Useful votes
                	reviewTotalVoteList.append(words[3]) #Total votes
                else:
                	reviewUsefulList.append(0)
                	reviewTotalVoteList.append(0)
                
            rating_alt = -1
            
            if len(img_tags) > 0:
            	for rating in img_tags:
            		alt = rating.get("alt")
            		if alt != None and alt.find("/") > 0:
            			rating_alt = alt[:alt.find("/")]
            			break		        				
            reviewRatingList.append(rating_alt)
    
        i = 0
        j = 0
        for text in text_tags:
            i = i+1
            if (i > 5) and (text.find(text=True) != None) and (text.find(text=True) != "*** This review may contain spoilers ***"):
                if j < len(reviewTitleList):
                    reviewList.append(cleanText(text.find(text=True)))          #Text
                    j = j+1 
            
        i = 0
        while i < len(reviewTitleList):
        	Review_List.append(Review(reviewTitleList[i], reviewAuthorList[i], reviewDateList[i], reviewRatingList[i], reviewUsefulList[i], reviewTotalVoteList[i], reviewList[i]))
        	i = i+1
        
    return Review_List;
    
def getMovieAttributes(all_movie_ids):
	movies = []
	reviewCnt = 0
	for id in all_movie_ids:
		title = ""
		rating = 0
		datePublished = ""
		year = ""
		directors = []
		actors = []
	
		movieLink = "http://www.imdb.com/title/" + id + "/"
		page = urllib2.urlopen(movieLink)
		soup = BeautifulSoup(page)
		
		#<meta content="Star Wars: Episode VII - The Force Awakens (2015)" property="og:title"/>
		temp = soup.findAll("meta", { "property" : "og:title" })
		if len(temp) > 0:
			title = temp[0].get("content")
			print("Id :" + id + " - Title :" + title)
		
		temp = soup.findAll("span", { "itemprop" : "ratingValue" })
		if len(temp) > 0:
			rating = temp[0].find(text = True)
		
		temp = soup.findAll("meta", { "itemprop" : "datePublished" })
		if len(temp) > 0:
			datePublished = temp[0].get("content")
			year = datePublished[:4]
		
		temp = soup.findAll("span", { "itemprop" : "director" })
		if len(temp) > 0:
			temp = temp[0].findAll("span", {"class" : "itemprop"})
			directors = []
			for i in temp:
				directors.append(i.find(text = True))
		
		temp = soup.findAll("span", { "itemprop" : "actors" })
		actors = []
		if len(temp) > 0:
			for i in temp:
				actorTemp = i.findAll("span", {"class" : "itemprop"})
				actors.append(actorTemp[0].find(text = True))
		
		reviews = []
		reviewLink = "http://www.imdb.com/title/" + id + "/reviews?start="
		reviews = getMovieReviews(reviewLink)
		
		if len(reviews) > 0:
			reviewCnt = reviewCnt + len(reviews)
		
		#movies.append(json.dumps(Movie(title, rating, datePublished, year, directors, actors, reviews), default=lambda o: o.__dict__))
		movies.append(Movie(title, rating, datePublished, year, directors, actors, reviews))
	print("No. of reviews: " + str(reviewCnt))
	return(movies)


all_movies = getMovieTitlesByGenres("action", 200)
#all_movie_reviews = getMovieAttributes(["tt1375666"])
all_movie_reviews = getMovieAttributes(all_movies)

json_data = simplejson.dumps(all_movie_reviews, indent=4, skipkeys=True, sort_keys=True, default=lambda o: o.__dict__)
fd = open('imdbMovieReviews3.txt', 'w')
fd.write(json_data)
fd.close()


