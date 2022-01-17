# %%
from data_preprocessing import *


class TreeNode:

    def __init__(self, feature=0, threshold=0, right_node=None, left_node=None, min_info_gain=-1):  # feature_index
        """
        Initializing a decision node and leaf node
        """
        self.feature = feature
        self.threshold = threshold
        self.right_node = right_node
        self.left_node = left_node
        self.min_info_gain = min_info_gain
        # value is the majority class for the leaf node. To determin the class of a datapoint
        self.value = None
        self.leaf_node_flag = False  # True only when it's a leaf node


class DecisionTreeClassifier:

    def __init__(self, max_depth=None, min_sample_split=None, num_features=None):  # max depth and minimum sample split. If number  of samples becomes less than min samples, we won't split any
        # further. This node will turn into a leaf node. When the tree reaches max depth we won't split any further either. These 2 variables can be == 2
        self.root = None
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.num_features = num_features


    def fit(self, x_train, y_train):
        if self.num_features: 
            self.num_features = min(x_train.shape[1], self.num_features)
        else:
            self.num_features = x_train.shape[1]  # creating range to randomize tree and avoid overfitting
        self.root = self.branchBuilder(x_train, y_train, curr_depth=0)  # init as zero


    def predict(self, x_test):
        return np.array([self.treeTraversal(x, self.root) for x in x_test])


    def treeTraversal(self, x, tree_node):
        if tree_node.leaf_node_flag:
            return tree_node.value

        if x[tree_node.feature] <= tree_node.threshold:
            return self.treeTraversal(x, tree_node.left_node)
        return self.treeTraversal(x, tree_node.right_node)

    def branchBuilder(self, x, y, curr_depth):  # init as zero
        samples, features = x.shape
        classes = np.unique(y).size

        if samples > self.min_sample_split and curr_depth < self.max_depth and classes > 1:  # pointless to compute if no unique target values
            feature_indexes = np.random.choice(features, self.num_features, replace=False)
            hyperparameters = self.HyperSplit(x, y, features)
            if hyperparameters["information_gain"] >= 0:
                left_idx, right_idx = self.nodeSplit(x[:, hyperparameters["feat_index"]], hyperparameters["threshold"])

                left_branch = self.branchBuilder(x[left_idx, :], y[left_idx], curr_depth + 1)
                right_branch = self.branchBuilder(x[right_idx, :], y[right_idx], curr_depth + 1)
                return TreeNode(hyperparameters["feat_index"], hyperparameters["threshold"], right_branch, left_branch, hyperparameters["information_gain"])
        return self.leafBuilder(y)

    def leafBuilder(self, y):
        leafNode = TreeNode()
        leafNode.leaf_node_flag = True
        y = y.tolist()
        leafNode.value = max(y, key=y.count)  # returns mode
        return leafNode


    def nodeSplit(self, x, _threshold):
        left_indeces = np.argwhere(x <= _threshold).flatten()
        right_indeces = np.argwhere(x > _threshold).flatten()
        return left_indeces, right_indeces

        left_indeces = tuple(np.transpose(left_indeces))
        right_indeces = tuple(np.transpose(right_indeces))
        return left_indeces, right_indeces

    def HyperSplit(self, x, y, features):
        info_gain = -1  # init with -1 because it can't be negative
        hyperparameters = {"information_gain": info_gain, "feat_index": 0, "threshold": 0, "left_idx": None, "right_idx":None}
        for idx in range(features):
            x_values = x[:, idx]
            thresholds = np.unique(x_values)
            for threshold in thresholds:
                left_idx, right_idx = self.nodeSplit(x_values, threshold)
                best_info_gain = self.InformationGainClassification(y, left_idx, right_idx, "entropy")  # seems to do better with gini
                if best_info_gain > info_gain:
                    hyperparameters["information_gain"] = best_info_gain
                    hyperparameters["feat_index"] = idx
                    hyperparameters["threshold"] = threshold
                    hyperparameters["left_idx"] = left_idx
                    hyperparameters["right_idx"] = right_idx
                    info_gain = best_info_gain
        return hyperparameters

    def giniIndex(self, child_node):  # 0 to 0.5
        """
        criterion for calculating IG. IG is used with decision trees to split a
        node because we measure the impurity of a node. If a node has multiple
        classes then it's impure. A one class node is pure.
        loss function: 1 - sigma(prob_data)**2
        """

        uniq_features = np.unique(child_node, return_counts=True, axis=0)[1]  # to get counts
        child_prob = uniq_features / child_node.size  # counting each class per node split
        return 1 - np.sum(child_prob**2)  # weighted gini index is for a particular split. I.e. gini index * (values of one side / divide by total between 2 sides)


    def Entropy(self, child_node):  # feed the vectors to this. 0 to 1. only x?
        """
        Balanced Probability Distribution (surprising): High entropy.
        sometimes it finds unique traits, other times it doesn't. too much info lost
        """

        uniq_features = np.unique(child_node, return_counts=True, axis=0)[1]  # to get counts
        data_prob = uniq_features / child_node.size  # vector size, value_counts --> node?
        return np.sum(-data_prob*np.log2(data_prob))
            #return np.sum(-dataset_prob*np.log2(dataset_prob+1e-9))  # entropy 

#In this way, entropy can be used as a calculation of the purity of a dataset, e.g.

    def InformationGainClassification(self, y, left_idx, right_idx, func):  # information gain is used with entropy. If it's used with the gini index, it's called something else
        # child vs parent nodes
        """
        information gain of a loss function
        y --> target
        mask --> split choice
        large IG means lower entropy
        """
        if left_idx.size == 0 or right_idx.size == 0:  # pure nodes, no need to divide any futhe
            return 0
        if func == "entropy":
            func = self.Entropy
        else:
            func = self.giniIndex
        left_node = y[left_idx]
        right_node = y[right_idx]
        left_inf_comp = left_node.size / y.size * func(left_node)
        right_inf_comp = right_node.size / y.size * func(right_node)
        return func(y) - (left_inf_comp + right_inf_comp)  # weighted values of features

myTree = DecisionTreeClassifier(max_depth=2, min_sample_split=2)
myTree.fit(X_train, Y_train)
prediction = myTree.predict(X_test)

