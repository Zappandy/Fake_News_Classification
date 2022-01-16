# %%
from data_preprocessing import *


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
        self.value = None

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


    def fit(self, x_train, y_train):
        #self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1]) num_features
        y_train = np.resize(y_train, (y_train.shape[0], 1))
        data = np.concatenate((x_train, y_train), axis=1)  # concatenating training data to grow tree and split node data that includes the labels
        self.root = self.branchBuilder(data, curr_depth=0)  # init as zero


    def predict(self, x_train):
        return np.array([self.treeTraversal(x, self.root) for x in x_train])


    def treeTraversal(self, x, tree_node):
        if tree_node.value:
            return tree_node.value

        print(tree_node.feature)
        if x[tree_node.feature] < node.threshold:
            return self.treeTraversal(x, tree_node.left)
        return self.treeTraversal(x, tree_node.right)

    def branchBuilder(self, data, curr_depth):  # init as zero
        x = data[:, :-1]
        y = data[:, -1]
        samples, features = x.shape
        if curr_depth > self.max_depth or samples < self.min_sample_split:
            leaf_node = TreeNode()
            y = y.tolist()
            leaf_node.value = max(y, key=y.count)  # returns mode
            return TreeNode(leaf_node)

        elif samples >= self.min_sample_split and curr_depth <= self.max_depth:
            hyperparameters = self.HyperSplit(data, x, samples, features)
            if hyperparameters["information_gain"] > 0:
                left_branch = self.branchBuilder(hyperparameters["left_branch"], curr_depth + 1)
                right_branch = self.branchBuilder(hyperparameters["right_branch"], curr_depth + 1)
                decision_node = TreeNode(hyperparameters["feat_index"], hyperparameters["threshold"], right_branch, left_branch, hyperparameters["information_gain"])
                return decision_node

            if hyperparameters["information_gain"] == 0:
                print("Pure node, each side corresponds to one class. No further splitting needed")
        else:
            raise Exception("Program did not enter the tree at all")

    def nodeSplit(self, data, _threshold):
        left_indeces = np.argwhere(data[:, :-1] <= _threshold)
        right_indeces = np.argwhere(data[:, :-1] > _threshold)
        left_indeces = tuple(np.transpose(left_indeces))
        right_indeces = tuple(np.transpose(right_indeces))
        left_node = data[left_indeces]
        right_node = data[right_indeces]
        parent = np.concatenate((left_node, right_node), axis=0)  # ignoring labels column
        return left_node, right_node, parent
        # https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/

    def HyperSplit(self, data, x, samples, features):
        info_gain = -1  # init with -1 because it can't be negative
        hyperparameters = {"information_gain": info_gain, "feat_index": 0, "threshold": None, "left_branch": None, "right_branch":None}
        for idx in range(features):
            x_values = x[:, idx]
            thresholds = np.unique(x_values)
            for threshold in thresholds:
                left_node, right_node, parent_node = self.nodeSplit(data, threshold)
                if bool(left_node.size) and bool(right_node.size):
                    best_info_gain = self.InformationGainClassification(left_node, right_node, parent_node, "entropy")  # seems to do better with gini
                    if best_info_gain > info_gain:
                        hyperparameters["information_gain"] = best_info_gain
                        hyperparameters["feat_index"] = idx
                        hyperparameters["threshold"] = threshold
                        hyperparameters["left_branch"] = left_node
                        hyperparameters["right_branch"] = right_node
                        info_gain = best_info_gain
                        print(best_info_gain)
        return hyperparameters

    def giniIndex(self, x):  # 0 to 0.5
        """
        criterion for calculating IG. IG is used with decision trees to split a
        node because we measure the impurity of a node. If a node has multiple
        classes then it's impure. A one class node is pure.
        loss function: 1 - sigma(prob_data)**2
        """

        uniq_features = np.unique(data, return_counts=True, axis=0)[1]  # to get counts
        x_prob = uniq_features / x.size  # counting each class per node split
        return 1 - np.sum(x_prob**2)  # weighted gini index is for a particular split. I.e. gini index * (values of one side / divide by total between 2 sides)


    def Entropy(self, child_node):  # feed the vectors to this. 0 to 1. only x?
        """
        Balanced Probability Distribution (surprising): High entropy.
        sometimes it finds unique traits, other times it doesn't. too much info lost
        """

        unique_features = np.unique(child_node)
        probabilities = []
        for uniq in unique_features:
            probabilities.append(len(child_node[child_node == uniq]) / child_node.size)
        probabilities = np.array(probabilities)
        
        #values, uniq_features = np.unique(child_node, return_counts=True, axis=0)  # to get counts
        #data_prob = uniq_features / child_node.size  # vector size, value_counts --> node?
        #print(np.sum(-data_prob*np.log2(data_prob)))
        #return np.sum(-data_prob*np.log2(data_prob))
        return np.sum(-probabilities*np.log2(probabilities))
            #return np.sum(-dataset_prob*np.log2(dataset_prob+1e-9))  # entropy 

#In this way, entropy can be used as a calculation of the purity of a dataset, e.g.

    def InformationGainClassification(self, left_child, right_child, parent, func):  # information gain is used with entropy. If it's used with the gini index, it's called something else
        # child vs parent nodes
        """
        information gain of a loss function
        y --> target
        mask --> split choice
        large IG means lower entropy
        """
        if func == "entropy":
            func = self.Entropy
        else:
            func = self.giniIndex
        left_inf_comp = left_child.size / parent.size * func(left_child)
        right_inf_comp = right_child.size / parent.size * func(right_child)
        return func(parent) - (left_inf_comp + right_inf_comp)  # weighted values of features

myTree = DecisionTreeClassifier(max_depth=2, min_sample_split=2)
myTree.fit(X_train, Y_train)
prediction = myTree.predict(X_test)

