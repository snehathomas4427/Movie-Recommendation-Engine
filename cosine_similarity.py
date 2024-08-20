from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

text = ["London Paris London", "Paris Paris London"]

cv = CountVectorizer() #CountVectorizer is a class from scikit learn
count_matrix = cv.fit_transform(text)
array = count_matrix.toarray()
print (array)

a = cosine_similarity(array)
print(a) #output: [[1.  0.8][0.8 1. ]]. means, text[0] is fully similar to text[0], text[0] is 0.8 similar to text[1], text[1] is 0.8 similar to text[0], text[1] is fully similar to text[1]

