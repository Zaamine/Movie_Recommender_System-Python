# Movie recommender System in Python
Repo for the Movie Recommender System implemented in Python which simulates an online interaction between a viewer and the platform, allowing him to get a recommendation of 10 movies according to his choices.

* Used the small version of the latest movie dataset created and indexed by MovieLens between 1995 and 2018: 100 836 ratings applied on 9742 movies by 610 users
* Step 1: Calculated the weighted average score between the popularity of a movie (its number of ratings) and the average of its ratings in order to propose to the viewer a choice of the 100 most popular films of the Cinema
* Step 2: Made a recommendation of 5 "popular" movies using a Machine Learning algorithm: k-Nearest Neighbors (kNN) with Scikit-learn
* Step 3: Made a recommendation of 5 "less known" movies using a second Machine Learning algorithm: Alternating Least Squares (ALS) with PySpark
* Step 3 bis: Alternative for the 5 "less known" movies recommended using a Deep Learning algorithm: Deep Neural Matrix Factorization (DNMF) with Tensorflow and Keras
* Deployed the final system using the pre-computed results from the precedent models on Flask
* Used the collaborative filtering method to predict the ratings made on a 5-star scale related to each possible (userId, movieId) couple of the dataset and obtaining an RMSE of 0.87

<!-- Articles related (written by me): [Medium](), [Towards Data Science]() -->

![](https://github.com/Zaamine/Zaamine/blob/main/images/recommender_system-screenshot_1.PNG)
![](https://github.com/Zaamine/Zaamine/blob/main/images/recommender_system-screenshot_2.PNG)

The idea behind the codes I uploaded in this repo is to show you what are the important aspects to take into account in order to concretely implement the machine learning and deep learning models used in my movie recommender system, so it helps you implement your very own version of it. It's not about just cloning the repo. The project presented here actually required 4 months of research and work during my internship in Data Science at Deutsche Telekom, so I only shared the main points.
