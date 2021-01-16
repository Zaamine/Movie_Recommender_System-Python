"""
This code is a condensed version of the step 2 of Amine Zaamoun's Movies Recommender System and was created in January 16, 2021.
It originally runs on Jupyter Notebook, thanks to which each code snippet can be broken down and analyzed in a different cell.

For this particular file to work, it must be linked to the code related to step 1 of the film recommendation system, available in the file "Weighted_Average_Score-Calculation.py".
"""

"""
importing the needed libraries
"""
from scipy.sparse import csr_matrix
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import pickle

"""
creating a pivot table
"""
movies_pivot = mostRatedMovies.groupBy('title').pivot('userId').sum('rating').fillna(0)
movie_features_df = movies_pivot.toPandas().set_index('title')
movie_features_df_matrix = csr_matrix(movie_features_df.values)

"""
splitting the "mostRatedMovies" dataset into a train and test sets
"""
X = mostRatedMovies.toPandas()[['userId', 'movieId']].values
y = mostRatedMovies.toPandas()['rating'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

"""
fitting the data to the supervised kNeighborsRegressor model (to get predictions of the ratings)
"""
knn_reg = KNeighborsRegressor(n_neighbors=11, n_jobs=-1) # the first neighbor of a movie is the movie itself, so we specify 11
knn_reg.fit(X_train, y_train)
y_pred = knn_reg.predict(X_test)

"""
evaluating the kNeighborsRegressor model
"""
# out-of-sample evaluation
rmse_test = sqrt(mean_squared_error(y_test, y_pred))
print("Out-of-sample RMSE = " + str(rmse_test))

# in-sample-evaluation
knn_reg_whole = KNeighborsRegressor(n_neighbors=11, n_jobs=-1)
knn_reg_whole.fit(X, y)
y_pred_whole = knn_reg_whole.predict(X)
rmse = sqrt(mean_squared_error(y, y_pred_whole))
print("Root-mean-square error = " + str(rmse))

"""
fitting the final unsupervised model NearestNeighbors to find the most similar movies of each ones using the whole dataset
"""
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=11, n_jobs=-1)
model_knn.fit(movie_features_df_matrix)

# choosing a title from our movie matrix
favoriteMovie = 'Iron Man (2008)'
query_index = movie_features_df.index.get_loc(favoriteMovie)
distances, indices = model_knn.kneighbors(movie_features_df.loc[favoriteMovie,:].values.reshape(1, -1), n_neighbors=11)

# printing the 10 most similar movies according to the kNN model
for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for {0}:\n'.format(movie_features_df.index[query_index]))
    else:
        print('{0}: {1}, with distance of {2}:'.format(i, movie_features_df.index[indices.flatten()[i]], distances.flatten()[i]))


# saving the table to CSV file for later access
movie_features_df.to_csv(os.path.join(trained_datapath, 'kNNmovieMatrix.csv'))

# saving the model to disk
file = open(os.path.join(trained_datapath, 'kNN_model.sav'), 'wb')
pickle.dump(model_knn, file)

# some time later... load the model from disk
file_to_load = open(os.path.join(trained_datapath, 'kNN_model.sav'), 'rb')
loaded_kNN_model = pickle.load(file_to_load)
