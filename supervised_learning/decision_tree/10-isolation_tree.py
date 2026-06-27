#!/usr/bin/env python3
'''Random Forest to detect outliers using Isolation Trees'''
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree():
    '''Isolation Random Tree Classifier'''
    def __init__(self, max_depth=10, seed=0, root=None):
        '''Initializes the isolation random tree classifier'''
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        '''Return a string representation of the tree.'''
        return self.root.__str__()

    def depth(self):
        '''Return the maximum depth of the tree.'''
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        '''Return the number of nodes in the tree.'''
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def get_leaves(self):
        '''Return a list of all leaves in the tree.'''
        return self.root.get_leaves_below()

    def update_bounds(self):
        '''Update the bounds of all nodes in the tree.'''
        self.root.update_bounds_below()

    def update_predict(self):
        '''Update the predict function of the tree.'''
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        def predict(A):
            '''Predict the class for each sample in A.'''
            result = np.zeros(A.shape[0], dtype=int)
            for leaf in leaves:
                mask = leaf.indicator(A)
                result[mask] = leaf.value
            return result

        self.predict = predict

    def np_extrema(self, arr):
        '''Return the minimum and maximum of a numpy array.'''
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        '''Return a random feature and threshold for splitting the data.'''
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            sub = self.explanatory[:, feature][node.sub_population]
            if sub.size <= 1:
                return feature, sub[0] if sub.size == 1 else 0
            feature_min, feature_max = self.np_extrema(sub)
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        '''Return a new leaf child of the given node'''
        leaf_child = Leaf(node.depth + 1)
        leaf_child.depth = node.depth+1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        '''Return a new node child of the given node'''
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        '''Fit the given node to the explanatory features.'''
        node.feature, node.threshold = self.random_split_criterion(node)

        mask = self.explanatory[:, node.feature] > node.threshold
        left_population = node.sub_population & mask
        right_population = node.sub_population & ~mask

        # Is left node a leaf ?
        is_left_leaf = (
            node.depth + 1 >= self.max_depth or
            np.sum(left_population) < self.min_pop
        )

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Is right node a leaf ?
        is_right_leaf = (
            node.depth + 1 >= self.max_depth or
            np.sum(right_population) < self.min_pop
        )

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """Fit the isolation tree to the explanatory features."""
        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        self.root.sub_population = np.ones(explanatory.shape[0], dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}""")
