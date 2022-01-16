# %%
from data_preprocessing import *
from sklearn.model_selection import train_test_split


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
        y_train = np.resize(y_train, (y_train.shape[0], 1))
        data = np.concatenate((x_train, y_train), axis=1)  # concatenating training data to grow tree and split node data that includes the labels
        self.root = self.branchBuilder(data, curr_depth=0)  # init as zero


    def predict(self, x_train, y_train):
        pass

    def branchBuilder(self, data, curr_depth):  # init as zero
        x = data[:, :-1]
        y = data[:, -1]
        samples, features = x.shape
        if curr_depth > self.max_depth or samples < self.min_sample_split:
            leaf_node = TreeNode()
            #y = y.tolist()
            #leaf_node.value = max(y, key=y.count)  # returns mode
            leaf_node.value = np.argmax(np.bincount(y))  # faster, no need to cast
            return TreeNode(leaf_node)

        elif samples >= self.min_sample_split and curr_depth <= self.max_depth:
            self.tree_split = self.HyperSplit(x, y, samples, features)
            if self.IG > 0:
                left_branch = self.branchBuilder(x, y, threshold, currdepth + 1)
                right_branch = self.branchBuilder
                decision_node =help
                TreeNode(decision_node)  # mess of values to pass or attributes?

            if self.IG == 0:
                print("Pure node, each side corresponds to one class. No further splitting needed")

    def nodeSplit(self, data, feature_index, split_threshold):
        # MAY NOT need feature_index, it's the col index
        left_node = np.argwhere(data <= split_threshold).flatten()  # find indexes turn 1d
        right_node = np.argwhere(data > split_threshold).flatten()  # find indexes turn 1d
        #right_node = np.array([row for row in data if row[feature_index] > split_threshold])  # slower
        print(left_node)
        print(right_node)
        raise SystemExit
        return left_node, right_node
        # https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/

    def HyperSplit(self, x, y, samples, features):
        #data = np.concatenate((x, y), axis=1)
        #print(x[:, 0].shape)
        #print(x[:, 0][0].shape)
        for idx in range(features):
            x_values = x[:, idx]
            thresholds = np.unique(x_values)
            for threshold in thresholds:
                self.nodeSplit(x, idx, threshold)


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

myTree = DecisionTreeClassifier(max_depth=4, min_sample_split=2)
myTree.fit(X_train, Y_train)
