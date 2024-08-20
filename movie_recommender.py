import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
###### helper functions. Use them when needed #######
def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]
##################################################

##Step 1: Read CSV File
df = pd.read_csv('movie_dataset.csv') #df stands for dataframe 
#print(df.head())

##Step 2: Select Features
features = ["keywords", "cast", "genres", "director"]

##Step 3: Create a column in DF which combines all selected features
for feature in features:
    df[feature] = df[feature].fillna('') #fillna('') replaces any missing values with an empty string 

def combine_features(row):
    try:
        return row["keywords"] + " " + row["cast"] + " " + row["genres"] + " " + row["director"]
    except Exception as e:
        print(f"Error in row: {row.name}, {e}")

df["combine_features"] = df.apply(combine_features, axis=1) #The apply function applies a function (combine_features) along the axis of a df. axis=1 means that the function is applied row-wise. If axis=0 -> column-wise.
#print("Combined Features:\n", df["combine_features"].head())

##Step 4: Create count matrix from this new combined column
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combine_features"])
#print(count_matrix) #the output "coords" is the (row, col) and the "value" is how many times that word appears in that ROW.

##Step 5: Compute the Cosine Similarity based on the count_matrix
similar = cosine_similarity(count_matrix)
#print(similar)

movie_user_likes = "Avatar"

## Step 6: Get index of this movie from its title
movie_index = get_index_from_title(movie_user_likes)
similar_movies = list(enumerate(similar[movie_index]))

## Step 7: Get a list of similar movies in descending order of similarity score
sorted_movies = sorted(similar_movies, key=lambda x:x[1], reverse=True)

## Step 8: Print titles of first 50 movies
i = 0
for movie in sorted_movies:
    print(get_title_from_index(movie[0]))
    i += 1
    if i > 50:
        break