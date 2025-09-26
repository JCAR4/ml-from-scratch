import numpy as np
m = 100
np.random.seed(42)
X_data=np.random.rand(m, 2)
y=np.array([1 if x[0] + x[1] > 1 else 0 for x in X_data])
#print(X_data[:5])
#print(y[:5])

def sigmoid(z):
    """
    Compute the sigmoid of z

    Parameters:
    z : float or np.array
        Input value or array

    Returns:
    sigmoid(z) : float or np.array
        Sigmoid of input
    """
    return 1 / (1 + np.exp(-z))

#print(sigmoid(0))  # Should print 0.5
#print(sigmoid(np.array([0, 1, 2])))  # Should print [0.5, 0.73105858, 0.88079708]

def compute_cost(X, y, w, b):
    """
    Compute the binary cross-entropy cost
    """
    m = X.shape[0]
    z = np.dot(X, w) + b
    y_hat = sigmoid(z)
    
    cost = - (1/m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    return cost
def compute_gradients(X, y, w, b):
    """
    Compute gradients of cost w.r.t weights and bias
    """
    m = X.shape[0]
    z = np.dot(X, w) + b
    y_hat = sigmoid(z)
    
    dw = (1/m) * np.dot(X.T, (y_hat - y))  # gradient for weights
    db = (1/m) * np.sum(y_hat - y)         # gradient for bias
    
    return dw, db

class LogisticRegression:
    def __init__(self, learning_rate=0.1, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.w = None
        self.b = None

    def fit(self, X, y):
        """
        Train the logistic regression model using gradient descent
        """
        n_features = X.shape[1]
        self.w = np.zeros(n_features)
        self.b = 0

        for i in range(self.num_iterations):
            # Compute gradients
            dw, db = compute_gradients(X, y, self.w, self.b)
            
            # Update weights and bias
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            
            # Optional: print cost every 100 iterations
            if i % 100 == 0:
                cost = compute_cost(X, y, self.w, self.b)
                print(f"Iteration {i}, Cost: {cost:.4f}")

    def predict_proba(self, X):
        """
        Compute probability estimates for input X
        """
        z = np.dot(X, self.w) + self.b
        return sigmoid(z)

    def predict(self, X):
        """
        Predict class labels (0 or 1) for input X
        """
        y_prob = self.predict_proba(X)
        return np.array([1 if p > 0.5 else 0 for p in y_prob])

# Initialize and train
model = LogisticRegression(learning_rate=0.5, num_iterations=1000)
model.fit(X_data, y)

# Make predictions
predictions = model.predict(X_data)

# Check accuracy
accuracy = np.mean(predictions == y)
print(f"Training Accuracy: {accuracy*100:.2f}%")
