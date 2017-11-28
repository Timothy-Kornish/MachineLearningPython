import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#-------------------------------------------------------------------------------
#                   Movie Recommender System
#-------------------------------------------------------------------------------


columns_names = ['user_id', 'item_id', 'rating', 'timestamp']

df = pd.read_csv('u.data', sep = '\t', names = columns_names)
movie_titles = pd.read_csv('Movie_Id_Titles.csv')

print('\n-------------------------------------------------------------------\n')
print("user data head:\n", df.head())
print('\n-------------------------------------------------------------------\n')
print("movie title data head:\n", movie_titles.head())
print('\n-------------------------------------------------------------------\n')

#-------------------------------------------------------------------------------
#       Merging two seperate data frames into one
#-------------------------------------------------------------------------------

df = pd.merge(df, movie_titles, on = 'item_id')
print("newly mergeed data head:\n", df.head())
print('\n-------------------------------------------------------------------\n')

sns.set_style("white")

print("average rating for every movie:\n",
       df.groupby('title')['rating'].mean().sort_values(ascending = False))
print('\n-------------------------------------------------------------------\n')

print("number of ratings for every movie:\n",
       df.groupby('title')['rating'].count().sort_values(ascending = False))
print('\n-------------------------------------------------------------------\n')

ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())

print("ratings data head:\n", ratings.head())
print('\n-------------------------------------------------------------------\n')


ratings['num of ratings'].hist(bins = 70)
plt.show()

ratings['rating'].hist(bins = 70)
plt.show()

sns.jointplot(x = 'rating', y = 'num of ratings', data = ratings, alpha = 0.5)
plt.show()

#-------------------------------------------------------------------------------
#       Finding correlation among movies based on user ratings
#-------------------------------------------------------------------------------

moviemat = df.pivot_table(index = 'user_id', columns = 'title', values = 'rating')

print("ratings of each movie for every viewer data head:\n", moviemat.head())
print('\n-------------------------------------------------------------------\n')

ratings.sort_values('num of ratings', ascending = False).head(10)

starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']

#-------------------------------------------------------------------------------
#       Finding correlation for star wars
#-------------------------------------------------------------------------------

similar_movies_to_starwars = moviemat.corrwith(starwars_user_ratings)
similar_movies_to_liarliar = moviemat.corrwith(liarliar_user_ratings)

corr_starwars = pd.DataFrame(similar_movies_to_starwars, columns = ['Correlation'])
corr_starwars.dropna(inplace = True)

print("Movies correlated to Star Wars based on Ratings alone data head:\n",
       corr_starwars.sort_values("Correlation", ascending = False))
print('\n-------------------------------------------------------------------\n')

corr_starwars = corr_starwars.join(ratings['num of ratings'])
starwars_head = corr_starwars[corr_starwars['num of ratings'] > 100].sort_values("Correlation", ascending = False)

print("Movies correlated to Star Wars with more than 100 ratings data head:\n",
       starwars_head.head())
print('\n-------------------------------------------------------------------\n')

#-------------------------------------------------------------------------------
#       Finding correlation for Liar Liar
#-------------------------------------------------------------------------------

corr_liarliar = pd.DataFrame(similar_movies_to_liarliar, columns = ['Correlation'])
corr_liarliar.dropna(inplace = True)

corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
liarliar_head = corr_liarliar[corr_liarliar['num of ratings'] > 100].sort_values("Correlation", ascending = False)

print("Movies correlated to Liar Liar with more than 100 ratings data head:\n",
       liarliar_head.head())
print('\n-------------------------------------------------------------------\n')
