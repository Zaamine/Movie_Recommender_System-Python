"""
This code is a snippet of the condensed version of Amine Zaamoun's deployed Movies Recommender System and was created in January 18, 2021.
This file doesn't work by itself in a Flask app, it needs to be at least associated to html templates.
The idea behind this file is just to show you how I used the pre-computed results of the already trained machine learning and deep learning models to give movie recommendations.
"""

# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
import pandas as pd
import os
import time
import random
import pickle
from functions import show_popular_movies, get_movieIds, get_movieTitles, get_moviePosters

app = Flask(__name__)

# set random seed to get reproductible results
random.seed(42)

# path configs
data_path = os.environ['DATA_PATH']
movies_datapath = os.path.join(data_path, 'Movies/MovieLens/movies_data-100k')
trained_datapath = os.path.join(movies_datapath, 'Already_Trained')

# load the movies metadataset (the new one created due to the reduction of the ratings dataset)
movies = pd.read_csv(
    os.path.join(trained_datapath, 'movies_new_augmented.csv'),
    usecols=['movieId', 'title', 'genres', 'poster'],
    dtype={'movieId': 'int32', 'title': 'object', 'genres': 'object', 'poster': 'object'})

# load the weighted average data
mostPopularMovies = pd.read_csv(
    os.path.join(trained_datapath, 'MostPopularMovies.csv'),
    usecols=['movieId', 'title', 'weighted_average'],
    dtype={'movieId': 'int32', 'title': 'object', 'weighted_average': 'float32'})

# load the pre-trained kNN_model from disk
kNN_model_to_load = open(os.path.join(trained_datapath, 'kNN_model.sav'), 'rb')
loaded_kNN_model = pickle.load(kNN_model_to_load)

# load the movie matrix
kNNmovieMatrix = pd.read_csv(os.path.join(trained_datapath, 'kNNmovieMatrix.csv')).set_index('title')

# load the pre_computed user recommendations for each movie predicted with the DNMF model
movieRecsConverter = {'userRecommendations': pd.eval, 'userRatings': pd.eval}
pdMovieRecs = pd.read_csv(os.path.join(trained_datapath, 'DNMF_UserRecommendationsForAllMovies.csv'), converters=movieRecsConverter)

# load the pre_computed movie recommendations for each user predicted with the DNMF model
userRecsConverter = {'movieRecommendations': pd.eval, 'movieRatings': pd.eval}
pdUserRecs = pd.read_csv(os.path.join(trained_datapath, 'DNMF_MovieRecommendationsForAllUsers.csv'), converters=userRecsConverter)

@app.route("/")
def home():
	return render_template('index.html', choose_message="Here is a list of the most popular movies in our database, please choose 3 :")

@app.route('/recommended_movies', methods=['POST'])
def make_recommendations():

	finalRecommendations = []

	popular_movies_list = show_popular_movies(mostPopularMovies)
	favorite_movieTitles = request.form.getlist('cb')
	favorite_ids = get_movieIds(movies, favorite_movieTitles)
	popular_movieIds_list = get_movieIds(movies, popular_movies_list)

	# for each favorite movies chosed by the user, add their 10 nearest neighbors to the kNN_recommendations list and keep only 5 randomly
	kNN_recommendations = []
	for i in range(3):
		userMovie = favorite_movieTitles[i]
		query_index = kNNmovieMatrix.index.get_loc(userMovie)
		distances, indices = loaded_kNN_model.kneighbors(kNNmovieMatrix.iloc[query_index,:].values.reshape(1, -1), n_neighbors = 11)
		for j in range(1, len(distances.flatten())):
			movieTitle = kNNmovieMatrix.index[indices.flatten()[j]]
			distance = distances.flatten()[j]
			if (movieTitle not in popular_movies_list) and (movieTitle not in kNN_recommendations) and (movieTitle not in favorite_movieTitles):
				kNN_recommendations.append(movieTitle)
	final_kNN_recommendations = random.sample(kNN_recommendations, 5)
	kNN_recommendedIds = get_movieIds(movies, final_kNN_recommendations)

	# for each favorites movies chosed by the user, add their top 5 recommended users to the DNMF_usersRecommendation list
	DNMF_usersRecommendation = []
	for i in range(3):
		movieChosed = favorite_ids[i]
		for j in range(5):
			userRecommended = pdMovieRecs[pdMovieRecs.movieId == movieChosed]["userRecommendations"].iloc[0][j]
			predictedMatch = pdMovieRecs[pdMovieRecs.movieId == movieChosed]["userRatings"].iloc[0][j]
			if (userRecommended not in DNMF_usersRecommendation):
				DNMF_usersRecommendation.append(userRecommended)
	# for each users recommended by the model, add their top 5 recommended movies to the DNMF_moviesRecommendation list and keep only 5 randomly
	DNMF_moviesRecommendation = []
	for i in range(len(DNMF_usersRecommendation)):
		userChosed = DNMF_usersRecommendation[i]
		for j in range(5):
			movieRecommended = pdUserRecs[pdUserRecs.userId == userChosed]["movieRecommendations"].iloc[0][j]
			predictedRating = pdUserRecs[pdUserRecs.userId == userChosed]["movieRatings"].iloc[0][j]
			if (movieRecommended not in DNMF_moviesRecommendation) and (movieRecommended not in kNN_recommendedIds) and (movieRecommended not in favorite_ids) and (movieRecommended not in popular_movieIds_list):
				DNMF_moviesRecommendation.append(movieRecommended)            
	final_DNMF_recommendations = random.sample(DNMF_moviesRecommendation, 5)
	recommendedMovieTitles = get_movieTitles(movies, final_DNMF_recommendations)

	# join the two lists, in order to give 5 movie recommendations from the kNN model and 5 movie recommendations from the DNMF model
	finalRecommendations = final_kNN_recommendations + recommendedMovieTitles
	recommendedMoviePosters = get_moviePosters(movies, finalRecommendations)

	return render_template('index.html',
		choose_message="Here is a list of the most popular movies in our database, please choose 3 :",
		favorite_movies_message="Your 3 favorite movies are :",
		favorite_movies_list=favorite_movieTitles,
		recommendations_message="We recommend you the following movies :",
		recommendations_list=finalRecommendations,
		recommendations_posters=recommendedMoviePosters)

if __name__ == "__main__":
    app.run(debug=True)
