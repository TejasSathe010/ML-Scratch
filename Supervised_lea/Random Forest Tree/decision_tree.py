import numpy as np
from collections import Counter


def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])


class Node:

    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:

    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, y):
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def display_tree(self, col_names):
        # print('# Inorder traversal','\n','Left -> Root -> Right')
        # print('RootNode is', col_names[self.root.feature], 'And Threshold is', self.root.threshold)
        # self.print_tree(self.root, col_names)
        # self.display(self.root, col_names)
        self.traverse(self.root, col_names)
        

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stopping criteria
        if (depth >= self.max_depth
                or n_labels == 1
                or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)
        # if(self.n_feats == 30):
        #     print(feat_idxs)

        # greedily select the best split according to information gain
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)
        
        # grow the children that result from the split
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        #print(best_feat, best_thresh)
        return Node(best_feat, best_thresh, left, right)

    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold
        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh):
        # parent loss
        parent_entropy = entropy(y)
        # print(parent_entropy)

        # generate split
        left_idxs, right_idxs = self._split(X_column, split_thresh)
        

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # compute the weighted avg. of the loss for the children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # information gain is difference in loss before vs. after split
        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column < split_thresh).flatten()
        right_idxs = np.argwhere(X_column >= split_thresh).flatten()
        return left_idxs, right_idxs

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    

    # Display Method 1 -->]

    # def print_tree(self, node, col_names):
    #     if node.left:
    #         self.print_tree(node.left, col_names)
    #     print(col_names[node.feature],'\t-->', node.threshold) if node.value is None else print(node.value),
    #     if node.right:
    #         self.print_tree(node.right, col_names)


    # Display Method 2 -->]

    # def display(self, Node, col_names):
    #     lines, _, _, _ = self._display_aux(Node, col_names)
    #     for line in lines:
    #         print(line)

    # def _display_aux(self, Node, col_names):
    #     """Returns list of strings, width, height, and horizontal coordinate of the root."""
    #     # No child.
    #     if Node.right is None and Node.left is None:
    #         line = '%s' % self.return_values(Node, col_names)
    #         width = len(line)
    #         height = 1
    #         middle = width // 2
    #         return [line], width, height, middle

    #     # Only left child.
    #     if Node.right is None:
    #         lines, n, p, x = self._display_aux(Node.left, col_names)
    #         s = '%s' % self.return_values(Node, col_names)
    #         u = len(s)
    #         first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
    #         second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
    #         shifted_lines = [line + u * ' ' for line in lines]
    #         return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

    #     # Only right child.
    #     if Node.left is None:
    #         lines, n, p, x = self._display_aux(Node.right, col_names)
    #         s = '%s' % self.return_values(Node, col_names)
    #         u = len(s)
    #         first_line = s + x * '_' + (n - x) * ' '
    #         second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
    #         shifted_lines = [u * ' ' + line for line in lines]
    #         return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

    #     # Two children.
    #     left, n, p, x = self._display_aux(Node.left, col_names)
    #     right, m, q, y = self._display_aux(Node.right, col_names)
    #     s = '%s' % self.return_values(Node, col_names)
    #     u = len(s)
    #     first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
    #     second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
    #     if p < q:
    #         left += [n * ' '] * (q - p)
    #     elif q < p:
    #         right += [m * ' '] * (p - q)
    #     zipped_lines = zip(left, right)
    #     lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
    #     return lines, n + m + u, max(p, q) + 2, n + u // 2


    # Display Method 3 -->]

    # def return_values(self, Node, col_names):
    #     if Node.value is None:
    #         return '%s=>%s' % (Node.feature, Node.threshold)
    #     return Node.value
    #     #(col_names[node.feature],'=>', node.threshold) if node.value is None else node.value)
    
    def traverse(self, root, col_names):
        current_level = [root]
        while current_level:
            print(' '.join(str(self.return_values(node, col_names)) for node in current_level))
            next_level = list()
            for n in current_level:
                if n.left:
                    next_level.append(n.left)
                if n.right:
                    next_level.append(n.right)
            current_level = next_level
    
    def return_values(self, Node, col_names):
        if Node.value is None:
            return '%s=>%s' % (col_names[Node.feature], Node.threshold)
        return Node.value
        #(col_names[node.feature],'=>', node.threshold) if node.value is None else node.value)
    
    


