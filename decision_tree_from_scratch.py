import numpy as np
from collections import Counter

# --- Node Class ---
class Node:
    """Represents a node in the decision tree."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # Feature index (int) to split on
        self.threshold = threshold  # Threshold value (float)
        self.left = left            # Left child node (Node object)
        self.right = right          # Right child node (Node object)
        self.value = value          # If leaf, stores the class label (int)

# --- Entropy and Information Gain Helper Functions (Global) ---

def entropy(y):
    """Calculates the entropy of a target array y."""
    # Ensure y is an array of integers (class labels)
    counts = Counter(y)
    total_samples = len(y)
    ent = 0.0
    for count in counts.values():
        p = count / total_samples
        # Add a small epsilon to avoid log(0) if needed, though p > 0 check handles it
        ent -= p * np.log2(p) if p > 0 else 0
    return ent

def information_gain(y, y_left, y_right):
    """Calculates the Information Gain from a split."""
    total_samples = len(y)
    if total_samples == 0:
        return 0.0

    p_left = len(y_left) / total_samples
    p_right = len(y_right) / total_samples

    # Information Gain = Entropy(Parent) - [w_left * Entropy(Left) + w_right * Entropy(Right)]
    return entropy(y) - (p_left * entropy(y_left) + p_right * entropy(y_right))

def best_split(X, y):
    """Finds the best feature and threshold to split the data based on maximum Information Gain."""
    n_samples, n_features = X.shape
    best_gain = -1
    split_feature, split_thresh = None, None

    # Iterate over every feature
    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])
        # Iterate over every unique value as a potential threshold
        for t in thresholds:
            # Create boolean masks for the split
            left_idx = X[:, feature] <= t
            right_idx = X[:, feature] > t

            # Skip split if it results in empty nodes
            if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
                continue

            gain = information_gain(y, y[left_idx], y[right_idx])
            
            if gain > best_gain:
                best_gain = gain
                split_feature = feature
                split_thresh = t

    return split_feature, split_thresh

# --- DecisionTree Class (Methods Integrated) ---
class DecisionTree:
    """A Decision Tree Classifier implementation using Information Gain (Entropy)."""
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _build_tree(self, X, y, depth=0):
        """Recursively builds the decision tree."""
        # 1. Stopping conditions
        if (len(np.unique(y)) == 1 or 
            depth >= self.max_depth or 
            len(y) < self.min_samples_split):
            
            # Find the most common class label for the leaf node
            # np.bincount(y) counts occurrences, .argmax() finds the index (the label) of the maximum count
            leaf_value = np.bincount(y).argmax()
            return Node(value=leaf_value)

        # 2. Find best split
        feature, thresh = best_split(X, y)
        
        # If no positive gain was found (best_split returns None)
        if feature is None:
            leaf_value = np.bincount(y).argmax()
            return Node(value=leaf_value)

        # 3. Split data
        left_idx = X[:, feature] <= thresh
        right_idx = X[:, feature] > thresh
        
        # 4. Recursively build child nodes
        left_node = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_node = self._build_tree(X[right_idx], y[right_idx], depth + 1)

        return Node(feature, thresh, left_node, right_node)

    def _traverse_tree(self, node, x):
        """Recursively traverses the tree to make a single prediction."""
        # If leaf node, return value
        if node.value is not None:
            return node.value
        
        # Decide which branch to follow based on feature value vs threshold
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(node.left, x)
        else:
            return self._traverse_tree(node.right, x)

    def fit(self, X, y):
        """Starts the tree building process by initializing the root node."""
        # Note: We call the internal _build_tree method
        self.root = self._build_tree(X, y)

    def predict(self, X):
        """Makes predictions for a list of samples X."""
        # Apply the traversal function to every sample in X
        return np.array([self._traverse_tree(self.root, x) for x in X])

# --- Execution ---

# Create synthetic dataset
np.random.seed(42)
# 20 samples, 2 features (X1, X2)
X_data = np.random.rand(20, 2) 
# Binary classification: 1 if X1 + X2 > 1, else 0
y = np.array([1 if x[0] + x[1] > 1 else 0 for x in X_data])

# Initialize and train
tree = DecisionTree(max_depth=3)
# The fit method is now correctly accessed as it belongs to the DecisionTree class
tree.fit(X_data, y)

# Make predictions
predictions = tree.predict(X_data)
print("Predictions:", predictions)
print("Actual     :", y)
accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy*100:.2f}%")
