"""
Here we define the logic of our 
web-application: 
a.k.a. here lives the Flask application
"""
import pandas as pd
from flask import Flask, render_template, request

import utils
from recommenders import random_recommender
from recommenders import nmf_recommender
from recommenders import cosim_recommender

# Flask main object that handles the web application and the server 
app = Flask(__name__)


@app.route("/")  # <- routing with decorator: mapping Url to what is been displayed on the screen
def landing_page():
    # return "Welcome to the Decisions recommender"
    return render_template("landing_page.html")


@app.route("/recommendation")
def recommendations_page():
    user_query = request.args.to_dict()
    movies = pd.read_csv('data/movies.csv')
    user_query = {movies.movieId[movies['title'] == movie]: int(rate) for movie, rate in user_query.items()}
    print(f'This is irritating{user_query}')
    # top4 = random_recommender(query=user_query, k=4)
    top4 = nmf_recommender(user_query, utils.nmf_model, k=4)
    print(top4)
    # return f"{top4}"
    # return render_template(
    #     "recommendations.html",
    #     movie1=top4[0],
    #     movie2=top4[1],
    #     movie3=top4[2],
    #     movie4=top4[3])
    return render_template(
        "recommendations.html",
        movie_list=top4)


if __name__ == "__main__":
    # It starts up the 
    # in-built development Flask server  
    app.run(debug=True, )
