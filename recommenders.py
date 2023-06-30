"""
Here lives our movie recommenders functions
"""
import random

import numpy as np
import pandas as pd

import utils
from utils import MOVIES, nmf_model, cosim_model
from scipy.sparse import csr_matrix

df = pd.read_csv('data/user_movie_df.csv', index_col=0)
df.drop(df[df.sum(axis=1)==0].index, inplace = True)
df_t =df.T


def nmf_recommender(query: dict, model=nmf_model, k=10) -> list:
    """
    Filters and recommends the top k movies for any given input query based on a trained NMF model.
    Returns a list of k movie ids.

    Args:
        query (dict): _description_
        model (_type_): _description_
        k (int, optional): _description_. Defaults to 10.

    Returns:
        list: topk movies
    """
    # 1. candidate generation
    # query - movieId and rating
    data = list(query.values())  # the ratings of the new user
    users = [0] * len(data)  # we use just a single row 0 for this user
    movies = list(query.keys())  # the columns (=movieId) of the ratings

    #print(model.n_features_in_)
    R_user = csr_matrix((data, (users, movies)), shape=(1, model.n_features_in_))

    # construct a user vector
    P = model.transform(R_user)
    Q = model.components_

    # 2. scoring
    scores = np.dot(P, Q)  # R_recommended is the score
    #print(scores)
    # The same result can be obtained by:
    # scores = model.inverse_transform(model.transform(R_user))  # R_recommended == scores

    # calculate the score with the NMF model
    scores = pd.Series(scores[0])

    # 3. ranking

    # filter out movies already seen by the user
    scores[query.keys()] = 0
    scores = scores.sort_values(ascending=False)
    recommendations = scores.head(10).index

    # return the top-k highest rated movie ids or titles

    return recommendations.values[:k]


def cosim_recommender(query: dict, model, k=10) -> list:
    """_summary_

    Args:
        query (dict): user query
        model (_type_): _description_
        k (int, optional): _description_. Defaults to 10.

    Returns:
        list: topk movies
    """
    return NotImplemented


def random_recommender(query={"Toy Story": 5}, k=3):
    """Toy random recommender

    Args:
        query (dict, optional): User query. Defaults to {"Toy Story": 5}.
        k (int, optional): _description_. Defaults to 3.

    Returns:
        _type_: _description_
    """
    random.shuffle(MOVIES)
    topk = MOVIES[:k]
    return topk


if __name__ == "__main__":
    #top3 = random_recommender()
    top2 = nmf_recommender({32: 5, 14: 5}, utils.nmf_model, k=2)
    #print(top3)
    print(top2)
