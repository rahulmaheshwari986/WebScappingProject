import urllib2
from bs4 import BeautifulSoup
import json

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
    reviewUserRating = -1
    reviewUseful = 0
    reviewTotalVote = 0
    reviewContent = ""
    
    def __init__(self, reviewTitle, reviewAuthor, reviewDate, reviewUserRating, reviewUseful, reviewTotalVote, reviewContent):
        self.reviewTitle = reviewTitle
        self.reviewAuthor = reviewAuthor
        self.reviewDate = reviewDate
        self.reviewUserRating = reviewUserRating
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
        
        content = review_soup.find("div", {'id' : 'tn15content'})
        
        text_tags = content.find_all("p")
        Reviews = content.find_all("div", {'class' : None, 'id' : None, 'style' : None})
        
        reviewTitleList = []
        reviewAuthorList = []
        reviewDateList = []
        reviewUserRatingList = []
        reviewUsefulList = []
        reviewTotalVoteList = []
        reviewList = []    

        for review in Reviews:            
    
            small_tags = review.find_all("small")
            anchor_tags = review.find_all("a")
            header_tags = review.find_all("h2")
            rating_tags = review.find_all("img")
            rating_alt = -1
            
            if len(small_tags) != 0:
                if len(rating_tags) > 0:
                    for rating in rating_tags:
                        alt = rating.get("alt")
                        if alt != None and alt.find("/") > 0:
                            rating_alt = alt[:alt.find("/")]
                            break                                
                reviewUserRatingList.append(rating_alt)                     #Rating
            
                reviewTitleList.append(header_tags[0].find(text=True))      #Title
                reviewAuthorList.append(anchor_tags[1].find(text=True))     #Author

                if(len(small_tags) == 3):                                   #Date 
                    reviewDateList.append(small_tags[2].find(text=True))
                elif len(small_tags) == 2:
                    reviewDateList.append(small_tags[1].find(text=True))
                else:
                    reviewDateList.append(small_tags[0].find(text=True))
                    
                #format of useful reviews "7 out of 11 people found the following review useful:"    
                if "found the following review useful" in small_tags[0].find(text=True):
                    words = small_tags[0].find(text=True).split()
                    reviewUsefulList.append(words[0])                       #Useful votes
                    reviewTotalVoteList.append(words[3])                    #Total votes
                else:
                    reviewUsefulList.append(0)
                    reviewTotalVoteList.append(0)
                                    
    
        i = 0
        for text in text_tags:
            if (text.find(text=True) != None) and (text.find(text=True) != "*** This review may contain spoilers ***"):
                if i < len(reviewTitleList):
                    reviewList.append(cleanText(text.find(text=True)))          #Text
                    i = i+1 
            
        i = 0
        while i < len(reviewTitleList):
            Review_List.append(Review(reviewTitleList[i], reviewAuthorList[i], reviewDateList[i], reviewUserRatingList[i], reviewUsefulList[i], reviewTotalVoteList[i], reviewList[i]))
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
        
        movies.append(json.dumps(Movie(cleanText(title), rating, datePublished, year, directors, actors, reviews), default=lambda o: o.__dict__))
    
    print("No. of reviews: " + str(reviewCnt))
    return(movies)


all_movies = getMovieTitlesByGenres("action", 100)
#all_movie_reviews = getMovieAttributes(["tt2094766"])
all_movie_reviews = getMovieAttributes(all_movies)

json_data = json.dumps(all_movie_reviews, indent=4, skipkeys=True, sort_keys=True, default=lambda o: o.__dict__)

fd = open('imdbMovieReviews1.txt', 'w')
fd.write(json_data)
fd.close()
