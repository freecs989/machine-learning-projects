import nltk
from nltk.corpus import movie_reviews	
import random


documents=[(list(movie_reviews.words(fileid)),category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

#print(documents[1])

all_words = []
for w in movie_reviews.words():
	all_words.append(w.lower())
print(len(all_words))
all_words = nltk.FreqDist(all_words)
#print(all_words.most_common())
word_features = list(all_words.keys())[:3000]

def find_features(documents):
	words = set(documents)
	features = {}
	for w in word_features:
		features[w] = (w in words)

	return features

featuresets = [(find_features(rev),category) for (rev,category) in documents]

train = featuresets[:2000]
test = featuresets[1800:]
print(len(featuresets))
classifier = nltk.NaiveBayesClassifier.train(train)
print("accuracy Naive Bayes",(nltk.classify.accuracy(classifier,test))*100)
classifier.show_most_informative_features(15)