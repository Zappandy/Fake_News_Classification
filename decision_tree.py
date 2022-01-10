import pandas as pd
import numpy as np

data = pd.read_csv("data.csv")
#data = pd.read_csv("vectors.csv", skipfooter=2000000, engine="python"))
data["obese"] = (data.Index >= 4).astype("int")
data.drop("Index", axis=1, inplace = True)
print(data.head())
print(data.columns)


# cost functions gin index  vs entropy 

def gini_index(x):
    """
    loss function: 1 - sigma(prob_data)**2
    """
    if isinstance(x, pd.Series):
        x_prob = x.value_counts() / x.shape[0]  # vector size
        #entropy = np.sum(-a*np.log2(a+1e-9))  
    return 1 - np.sum(x_prob**2)

def entropy(x):
    """
    loss function: sigma(-x_prob * log2(x_prob))
    """
    if isinstance(x, pd.Series):
        x_prob = x.value_counts() / x.shape[0]  # vector size
        return np.sum(-x_prob*np.log2(x_prob))  # entropy 
        #return np.sum(-x_prob*np.log2(x_prob+1e-9))  # entropy 
# information gain for classification

#https://www.section.io/engineering-education/entropy-information-gain-machine-learning/

# y --> target

#https://www.youtube.com/watch?v=sgQAhG5Q7iY
#TODO: x, y, GOTTA FIX THIS. Should be happening with x_Train, and y_Train
def information_gain_classification(x, total_set):  # information gain better with entropy than gini

    """
    information gain of a loss function
    y --> target
    mask --> split choice
    """
    entropy(total_set) - np.sum(x/total_set * entropy(x))

# entropy --> https://www.youtube.com/watch?v=y6VwIcZAUkI


# https://anderfernandez.com/en/blog/code-decision-tree-python-from-scratch/
# good comments on how DT work under the drop Index cell
