"""
Here live other stuff, e.g., unpickled model, movies
"""
import pickle

MOVIES = [
    "Tarzan",
    "300",
    "Dumbo",
    "Alice in Wonderland",
    "12 angry man",
    "Kunfu Panda 🐼"
]

# with open(yourmodelfile,rb) as file: 
#   nmf_model = pickle.load(file)
with open('models/nmf_recommender.pkl', 'rb') as file:
    nmf_model = pickle.load(file)
print(nmf_model)

cosim_model = ...