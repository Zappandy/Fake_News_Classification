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
        # threshold is important for splitting in the decision tree itself

        # value is the majority class for the leaf node. To determin the class of a datapoint
        # if a decision has been made
        self.value = None


class DecisionTreeClassifier:

    def __init__(self, max_depth=None, min_sample_split=None):  # max depth and minimum sample split. If number  of samples becomes less than min samples, we won't split any
        # further. This node will turn into a leaf node. When the tree reaches max depth we won't split any further either. These 2 variables can be == 2
        self.root = None
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split


    def fit(self, x_train, y_train):
        #self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1]) num_features
        #y_train = np.resize(y_train, (y_train.shape[0], 1))
        self.root = self.branchBuilder(x_train, y_train, curr_depth=0)  # init as zero


    def predict(self, x_test):
        return np.array([self.treeTraversal(x, self.root) for x in x_test])


    def treeTraversal(self, x, tree_node):
        if tree_node.value:
            return tree_node.value
        #print(tree_node.threshold)

        if x[tree_node.feature] <= tree_node.threshold:
            return self.treeTraversal(x, tree_node.left_node)
        return self.treeTraversal(x, tree_node.right_node)

    def branchBuilder(self, x, y, curr_depth):  # init as zero
        node = TreeNode()
        if len(x.shape) > 2:  # remove 1 dimensions from indexing
            x = np.squeeze(x) 
        #if len(x.shape) == 1:
        #    x = np.resize(x, (x.shape[0], 1))
        #    print("tito!!!")
        samples = x.shape[0]
        features = x.shape[1] if len(x.shape) > 1 else None
        print(x.shape[0], x.shape[1])

        if samples >= self.min_sample_split and curr_depth <= self.max_depth and features:  # pointless to compute if no unique target values
            hyperparameters = self.HyperSplit(x, y, samples, features)
            if hyperparameters["information_gain"] >= 0:
                left_idx, right_idx = self.nodeSplit(x[:, hyperparameters["feat_index"]], hyperparameters["threshold"])

                left_branch = self.branchBuilder(x[left_idx, :], y[left_idx], curr_depth + 1)
                right_branch = self.branchBuilder(x[right_idx, :], y[right_idx], curr_depth + 1)
                node = TreeNode(hyperparameters["feat_index"], hyperparameters["threshold"], right_branch, left_branch, hyperparameters["information_gain"])
                return node
        return self.leafBuilder(y, node)

    def leafBuilder(self, y, node):
        leaf_node = TreeNode()
        y = y.tolist()
        leaf_node.value = max(y, key=y.count)  # returns mode
        return TreeNode(leaf_node)


    def nodeSplit(self, x, _threshold):
        left_indeces = np.argwhere(x <= _threshold)
        right_indeces = np.argwhere(x > _threshold)
        print(np.transpose(left_indeces.shape))
        print(np.transpose(right_indeces.shape))
        raise SystemExit
        left_indeces = tuple(np.transpose(left_indeces))
        right_indeces = tuple(np.transpose(right_indeces))
        return left_indeces, right_indeces
        # https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/

    def HyperSplit(self, x, y, samples, features):
        info_gain = -1  # init with -1 because it can't be negative
        hyperparameters = {"information_gain": info_gain, "feat_index": 0, "threshold": None, "left_branch": None, "right_branch":None}
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
                    #hyperparameters["left_branch"] = left_node
                    #hyperparameters["right_branch"] = right_node
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

    def InformationGainClassification(self, y, left_idx, right_idx, func):  # information gain is used with entropy. If it's used with the gini index, it's called something else
        # child vs parent nodes
        """
        information gain of a loss function
        y --> target
        mask --> split choice
        large IG means lower entropy
        """
        if len(left_idx)== 0 or len(right_idx) == 0:  # pure nodes, no need to divide any futhe
            return 0
        left_node = y[left_idx]
        right_node = y[right_idx]
        #parent = np.concatenate((left_node, right_node), axis=0)  # ignoring labels column
        parent =y
        if func == "entropy":
            func = self.Entropy
        else:
            func = self.giniIndex
        left_inf_comp = left_node.size / parent.size * func(left_node)
        right_inf_comp = right_node.size / parent.size * func(right_node)
        return func(parent) - (left_inf_comp + right_inf_comp)  # weighted values of features

myTree = DecisionTreeClassifier(max_depth=2, min_sample_split=2)
myTree.fit(X_train, Y_train)
#prediction = myTree.predict(X_test)

