# %%
from data_preprocessing import *
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(data[:, :-1], data[:, -1], train_size=0.8, test_size=0.2)
X_train, X_valid, Y_train, Y_valid = train_test_split(data[:, :-1], data[:, -1], train_size=0.8, test_size=0.2)

class TreeNode:

    def __init__(self, feature=None, threshold=None, right_node=None, left_node=None, min_info_gain=None):  # feature_index
        """
        Initializing a decision node and leaf node
        """
        self.feature = feature
        self.threshold = threshold
        self.righ_node = right_node
        self.left_node = left_node
        self.min_info_gain = min_info_gain
        # threshold is important for splitting in the decision tree itself

        # value is the majority class for the leaf node. To determin the class of a datapoint
        # if a decision has been made
        self.value = value

    @property
    def leafNode(self):
        return self.value

    @leafNode.setter
    def leafNode(self, value):
        if value:
            self.value = value


class DecisionTreeClassifier:

    def __init__(self, max_depth=None, min_sample_split=None):  # max depth and minimum sample split. If number  of samples becomes less than min samples, we won't split any
        # further. This node will turn into a leaf node. When the tree reaches max depth we won't split any further either. These 2 variables can be == 2
        self.root = None
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.features = None


    def fit(self, x_train, y_train):
        #self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1]) num_features
        
        self.root = self.branchBuilder(x_train, y_train, curr_depth=0)  # init as zero


    def predict(self, x_train, y_train):
        pass

    def branchBuilder(self, x, y, curr_depth):  # init as zero
        samples = x.shape[0]
        features = x.size
        print(x[0][0].shape)
        #print(features)
        
        raise SystemExit

        if curr_depth > self.max_depth:
            y = y.tolist()
            max_value = max(y, key=y.count)
            leaf_node = TreeNode()
            leaf_node.value = max_value
            return TreeNode(leaf_node)


        if curr_samples >= self.min_sample_split and curr_depth <= self.max_depth:
            self.tree_split = self.HyperSplit()
            if self.IG > 0:
                left_branch = self.branchBuilder(x, y, threshold, xxxx, currdepth + 1)
                right_branch = self.branchBuilder
                decision_node =help
                TreeNode(decision_node)  # mess of values to pass or attributes?

            if self.IG == 0:
                print("Pure node, each side corresponds to one class. No further splitting needed")

    def nodeSplit(self, data, feature_index, split_threshold):
        # MAY NOT need feature_index
        left_node = np.argwhere(data <= splithreshold).flatten()  # find indexes turn 1d
        right_node = np.array([row for row in data if row[feature_index] > threshold])
        return left_node, right_node
        # https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/

    def HyperSplit(self, data):
         what


    def giniIndex(x):  # 0 to 0.5
        """
        criterion for calculating IG. IG is used with decision trees to split a
        node because we measure the impurity of a node. If a node has multiple
        classes then it's impure. A one class node is pure.
        loss function: 1 - sigma(prob_data)**2
        """
        if isinstance(x, pd.Series):
            x_prob = x.value_counts() / x.shape[0]  # counting each class per node split
        return 1 - np.sum(x_prob**2)  # weighted gini index is for a particular split. I.e. gini index * (values of one side / divide by total between 2 sides)




    def Entropy(data):  # feed the vectors to this. 0 to 1. only x?
        """
        Balanced Probability Distribution (surprising): High entropy.
        """
        dataset_prob = dataset.value_counts() / dataset.shape[0]  # vector size, value_counts --> node?
        return np.sum(-dataset_prob*np.log2(dataset_prob))  # entropy, this is fine
            #return np.sum(-dataset_prob*np.log2(dataset_prob+1e-9))  # entropy 

#In this way, entropy can be used as a calculation of the purity of a dataset, e.g.



    def InformationGainClassification(left_child, right_child, parent, func):  # information gain is used with entropy. If it's used with the gini index, it's called something else
        # child vs parent nodes
        """
        information gain of a loss function
        y --> target
        mask --> split choice
        large IG means lower entropy
        """
        left_inf_comp = left_child.size / parent.size * func(l_child)
        right_inf_comp = right_child.size / parent.size * func(r_child)
        return func(parent) - (left_inf_comp + right_inf_comp)  # weighted values of features
    # for regression
    #y.var() - np.sum(x.abs() / y.abs() * x.var())
    # IG(S, a) = H(S) - H(S|a), the second part is the cond probability. Parent node - comb entropy of childnodes
    # mutual information I(X;Y) = H(X) - H(X|Y) this is mutual dependence

myTree = DecisionTreeClassifier(2, 2)
myTree.fit(X_train, Y_train)
