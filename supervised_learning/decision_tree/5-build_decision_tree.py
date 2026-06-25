#!/usr/bin/env python3
'''Beginning of a decision tree implementation.'''
import numpy as np


class Node:
    '''A node in a decision tree.'''
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        '''Initialize a node in the decision tree.'''
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def __str__(self):
        '''Return a string representation of the node using prefixes'''
        if self.is_leaf:
            return f"-> leaf [value={self.value}]"
        elif self.is_root:
            left_str = self.left_child.__str__()
            right_str = self.right_child.__str__()
            left_str = self.left_child_add_prefix(left_str)
            right_str = self.right_child_add_prefix(right_str)
            return (f"root [feature={self.feature},"
                    f" threshold={self.threshold}]\n"
                    f"{left_str}{right_str}")
        else:
            left_str = self.left_child.__str__()
            right_str = self.right_child.__str__()
            left_str = self.left_child_add_prefix(left_str)
            right_str = self.right_child_add_prefix(right_str)
            return (f"-> node [feature={self.feature},"
                    f" threshold={self.threshold}]\n"
                    f"{left_str}{right_str}")

    def right_child_add_prefix(self, text):
        '''Add a prefix to the right child of this node.'''
        lines = text.rstrip("\n").split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       "+x) + "\n"
        return new_text

    def left_child_add_prefix(self, text):
        '''Add a prefix to the left child of this node.'''
        lines = text.rstrip("\n").split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  "+x) + "\n"
        return new_text

    def max_depth_below(self):
        '''Return the maximum depth of the tree below this node.'''
        if self.is_leaf:
            return self.depth
        else:
            left_depth = self.left_child.max_depth_below()
            right_depth = self.right_child.max_depth_below()
            return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        '''Return the number of nodes below this node.'''
        if self.is_leaf:
            return 1
        else:
            left_count = (self.left_child.count_nodes_below
                          (only_leaves=only_leaves))
            right_count = (self.right_child.count_nodes_below
                           (only_leaves=only_leaves))
            if only_leaves:
                return left_count + right_count
            else:
                return 1 + left_count + right_count

    def get_leaves_below(self):
        '''Return a list of all leaves below this node.'''
        if self.is_leaf:
            return [self]
        else:
            left_leaves = self.left_child.get_leaves_below()
            right_leaves = self.right_child.get_leaves_below()
            return left_leaves + right_leaves

    def update_bounds_below(self):
        '''Update the bounds of all nodes below this node.'''
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1*np.inf}

        for child in [self.left_child, self.right_child]:
            child.lower = self.lower.copy()
            child.upper = self.upper.copy()

            feature = self.feature
            threshold = self.threshold

            if child is self.left_child:
                # LEFT → feature > threshold
                if feature in child.lower:
                    child.lower[feature] = max(child.lower[feature], threshold)
                else:
                    child.lower[feature] = threshold

            else:
                # RIGHT → feature ≤ threshold
                if feature in child.upper:
                    child.upper[feature] = min(child.upper[feature], threshold)
                else:
                    child.upper[feature] = threshold

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):

        def is_large_enough(x):
            return np.all(
                np.array([np.greater(x[:, key], self.lower[key])
                          for key in self.lower.keys()]),
                axis=0
            )

        def is_small_enough(x):
            return np.all(
                np.array([np.less_equal(x[:, key], self.upper[key])
                          for key in self.upper.keys()]),
                axis=0
            )

        self.indicator = lambda x: np.all(np.array(
            [is_large_enough(x), is_small_enough(x)]), axis=0)


class Leaf(Node):
    '''A leaf node in a decision tree.'''
    def __init__(self, value, depth=None):
        '''Initialize a leaf node in the decision tree.'''
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def __str__(self):
        '''Return a string representation of the leaf node.'''
        return (f"-> leaf [value={self.value}]")

    def max_depth_below(self):
        '''Return the maximum depth of the tree below this node.'''
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        '''Return the number of nodes below this node.'''
        return 1

    def get_leaves_below(self):
        '''Return a list of all leaves below this node.'''
        return [self]

    def update_bounds_below(self):
        '''Update the bounds of all nodes below this node.'''
        pass


class Decision_Tree():
    '''A decision tree classifier.'''
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        '''Initialize a decision tree classifier.'''
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

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
