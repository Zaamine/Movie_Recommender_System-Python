"""
This code is a snippet of the condensed version of the step 1 of Amine Zaamoun's Movies Recommender System and was created in January 16, 2021.
It originally runs on Jupyter Notebook, thanks to which each code snippet can be broken down and analyzed in a different cell.
"""

"""importing the needed libraries"""
import os
from pyspark.sql.functions import mean, col

"""path config"""
data_path = os.environ['DATA_PATH']
movies_datapath = os.path.join(data_path, 'Movies/MovieLens/movies_data-100k')

"""loading the datasets"""
ratings = spark.read.load(os.path.join(movies_datapath, 'ratings.csv'), format='csv', header=True, inferSchema=True).drop("timestamp")
movies = spark.read.load(os.path.join(movies_datapath, 'movies.csv'), format='csv', header=True, inferSchema=True)

"""calculating the average rating and number of ratings of each movie"""
df = ratings.join(movies, on="movieId")
number_ratings = df.groupBy('movieId').count()
average_ratings = df.groupBy('movieId').avg('rating')
df_ratings = average_ratings.join(number_ratings, on="movieId")
df = df.join(df_ratings, on="movieId")
mostRatedMovies = df.where("count >= 50")

"""calculating the weighted average score of each movie"""
# We have to convert the 'vote_count' column which is a string type to a double type (numerical) in order to calculate the quantile
changedTypedf = mostRatedMovies.withColumn("vote_count", df["count"].cast("double"))
quantile_df = changedTypedf.approxQuantile("count", [0.75], 0)
m = quantile_df[0]

# collect() is used to return all the elements of the dataset as an array at the driver program. This is usually useful after a filter
# or other operation that returns a sufficiently small subset of the data.
mean_df = mostRatedMovies.select(mean(col('avg(rating)')).alias('mean')).collect()
C = mean_df[0]['mean']

movies_cleaned_df = mostRatedMovies.withColumn("weighted_average", ((mostRatedMovies['avg(rating)']*mostRatedMovies['count']) + (C*m)) / (mostRatedMovies['count']+m))

"""saving the table in CSV file for later access"""
trained_datapath = os.path.join(movies_datapath, 'Already_Trained')
movies_cleaned_pd.to_csv(os.path.join(trained_datapath, 'MostPopularMovies.csv'), index=False)
