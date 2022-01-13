import pandas as pd
import numpy as np

data = pd.read_csv("data.csv")
#data = pd.read_csv("vectors.csv", skipfooter=2000000, engine="python"))
data["obese"] = (data.Index >= 4).astype("int")
data.drop("Index", axis=1, inplace = True)
print(data.head())
print(data.columns)


# cost functions gin index  vs entropy 

def gini_index(x):  # 0 to 0.5
    """
    criterion for calculating IG. IG is used with decision trees to split a
    node because we measure the impurity of a node. If a node has multiple
    classes then it's impure. A one class node is pure.
    loss function: 1 - sigma(prob_data)**2
    """
    if isinstance(x, pd.Series):
        x_prob = x.value_counts() / x.shape[0]  # counting each class per node split
    return 1 - np.sum(x_prob**2)  # weighted gini index is for a particular split. I.e. gini index * (values of one side / divide by total between 2 sides)
# if one side has only one of the y targets, then the gini index would be 0. Having both is important. Here we see an actual data distribution
# the high gin index will be selected as the tree splitting criterion. Higher dependence, a balanced prob distribution
def entropy(data):  # feed the vectors to this. 0 to 1
    """
    loss function: sigma(-x_prob * log2(x_prob))
    logs of base 2 as per usual, but if using euler then we refer to the result as nats
    Shannon information --> self-information. -log p(x). Ensuring that the result is always
    negative or zero. If there is no surprise, certainty --> prob then 1 and thus low info
    information is in bits

    entropy for random variables and this is an example of Shannon information.
    The intuition for entropy is that it is the average number of bits required
    to represent or transmit an event drawn from the probability distribution for the random variable.

    Entropy for a random variable X with k in K discrete states turns into
    -sigma(each prob(k) in K set * the log of a prob of an event k

    lowest entropy for a random k variable is with probs == 1, certain.
    largest entropy is if all events are equally likely. An example of this is
    the shannon information from rolling a die vs the entropy of rolling a die

    This follows through with probability distributions

    Skewed Probability Distribution (unsurprising): Low entropy.
    Balanced Probability Distribution (surprising): High entropy.
    """
    if isinstance(dataset, pd.Series):  # scipy defaults to e
        # value_counts with bins in it can force a continuous variable into
        # a discrete one
        dataset_prob = dataset.value_counts() / dataset.shape[0]  # vector size, value_counts --> node?
        return np.sum(-dataset_prob*np.log2(dataset_prob))  # entropy, this is fine
        #return np.sum(-dataset_prob*np.log2(dataset_prob+1e-9))  # entropy 

#In this way, entropy can be used as a calculation of the purity of a dataset, e.g.
#how balanced the distribution of classes happens to be.
# IG provides a way to use entropy to calculate how changes in the dataset impact its purity
# this is the distribution of classes. Less surprise means more purity and that is smaller entropy

#TODO: x, y, GOTTA FIX THIS. Should be happening with x_Train, and y_Train
def information_gain_classification(x, y):  # information gain better with entropy than gini. x and y may be the nodes instead
    # child vs parent nodes
    """
    information gain of a loss function
    y --> target
    mask --> split choice
    computes the reduction in entropy or surprise from transforming a dataset
    a certain way. Evaluates the IG for each variable and chooses the variable that maximizes
    the information gain, thus minimizing the entropy, and 'best splits the dataset into groups
    for effective classification

    it can also be mutual information, which is a feature selection method. The gain of each variable
    is evaluated within the context of the target variable

    The first use is more common in the training of decision trees. IG is computed by comparing
    entropy of the dataset before and after transformation

    Mutual information calculates the statistical dependence between two variables and is the 
    name given to information gain when applied to variable selection.

    large IG means lower entropy
    """
    entropy(y) - np.sum(x/y * entropy(x))  # this is probably wrong, but this is
    # IG(S, a) = H(S) - H(S|a), the second part is the cond probability
    # mutual information I(X;Y) = H(X) - H(X|Y) this is mutual dependence
    # mutual information can also be seen as the Kullback-Leibler or KL. If the calculated result
    # is 0 the variables are independent. High results indicate higher dependence. Often used as a 
    # general form of a correlation coefficient. This is a measure of dependence between random
    # variables
    # this is why we see it in ICA to see whether components are statistically
    # independent
    # Effect of Transforms to a Dataset (decision trees): Information Gain.
    # Dependence Between Variables (feature selection): Mutual Information.

# https://machinelearningmastery.com/information-gain-and-mutual-information/ to read exmples 
# as to why IG in decision trees. The ID3 algorithm is used to build a decision tree
# The information gain is calculated for each variable in the dataset. The variable that has
# the largest information gain is selected to split the dataset. Generally, a larger gain
# indicates a smaller entropy or less surprise.
# https://www.amazon.com/Learning-McGraw-Hill-International-Editions-Computer/dp/0071154671/ref=as_li_ss_tl?keywords=machine+learning&qid=1563151324&s=books&sr=1-44&linkCode=sl1&tag=inspiredalgor-20&linkId=5488f209b18fb6b8ad8ce7f72d1c3ac5&language=en_US

# Information gain is the entropy of parent node minus sum of weighted entropies of all child nodes
# https://thatascience.com/learn-machine-learning/gini-entropy/


def infogain(parent, children, criterion):
    score = {'gini': giniscore, 'entropy': entropyscore}
    metric = score[criterion]
    parentscore = metric(parent)
    parentsum = sum(parent.values())
    weighted_child_score = sum([metric(i)*sum(i.values())/parentsum  for i in children])
    gain = round((parentscore - weighted_child_score),2)
    print('Information gain: {}'.format(gain))
    return gain

# https://machinelearningmastery.com/what-is-information-entropy/
# 


# checking purity of dataset using gini index https://ekamperi.github.io/machine%20learning/2021/04/13/gini-index-vs-entropy-decision-trees.html

