#experiments in OOP
"""
Programmed linear regression in r using functions but had to modify a complex structure to run code
now I will use OOP to fit linear regression to fit a polynomial the sin(x) function

shapes of inputs
x : n x p
w : p x 1
b : 1 x 1
y : n : 1

"""
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression():
    def __init__(self, init_w, init_b, alpha, lmd):
        self.alpha = alpha
        self.lmd = lmd
        self.w = init_w
        self.b = init_b
        self.history = []
    
    def predict(self, X):
        return(self.b + X @ self.w)
    
    #why cost in regression and loss in classification?
    def cost(self, X, y):
        n, _ = y.shape
        mse = self.mse(X, self.w, self.b, y)
        reg = (self.lmd / (2 * n)) + np.dot(self.w.T, self.w)
        return mse + reg
    
    def mse(self, X, y):
        n, _ = y.shape
        error = self.predict(X) - y
        return np.sum((error**2) / n)
    
    def gradients(self, X, y):
        n, _ = y.shape
        error = self.predict(X) - y # nx1
        dw = (X.T @ error + self.lmd * self.w) / n
        db = np.sum(error) / n
        return dw, db
    
    def train(self, X, y, epochs):
        for i in range(epochs):
            dw, db = self.gradients(X, y)
            self.w = self.w - self.alpha * dw
            self.b = self.b - self.alpha * db
            
            if i % 10 == 0:
                self.history.append(self.mse(X, y))
        print(f"model successfully trained for {epochs} epochs")
        print(self.w, '\n', self.b)
    
    def plot(self, X, y):
        # Plot results
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot predictions
        preds = self.predict(X)
        axes[0].scatter(range(len(preds)), preds, label="Predicted", color="red")
        axes[0].set_title("Predictions")

        # Plot original sin curve
        axes[1].scatter(range(len(y)), y, label="Original", color="blue")
        axes[1].set_title("Original Sine Curve")

        #plot history
        axes[2].plot(np.array(range(len(self.history))) * 10, self.history)
        axes[2].set_title("Training Loss History")
        plt.show()


x = np.arange(-40, 41) / 10
y = np.array(np.sin(x), ndmin = 2).T

#original r expoeriment on sine
#reflected on the importance of scaling features and noticed no affect of lambda, because no feature is noise

p1 = 12
X1 = np.column_stack([x**i for i in range(1, p1 + 1)])
X1 = (X1 - np.mean(X1, axis = 0))/np.std(X1, axis=0)

w = np.zeros((p1, 1))
b = 0
m1 = LinearRegression(w, b, 0.1, 0.1)

m1.train(X1, y, epochs = 1000)
predictions = m1.predict(X1)
m1.plot(X1, y)


#adding noisy features such as x = constant, x = cos(x), x = e^x

p2 = 12
X2 = np.column_stack([x**i for i in range(1, p2 + 1)])
_1 = np.column_stack([np.cos(x)**i for i in range(1, p2 + 1)])
_2 = np.column_stack([np.cos(x)**i for i in range(1, p2 + 1)])
_3 = np.column_stack([np.zeros(y.shape) + i for i in range(1, p2 + 1)])
_4 = np.column_stack([np.zeros(y.shape) + 10 for i in range(1, p2 + 1)])
_5 = np.column_stack([np.exp(x) ** i for i in range(1, p2 + 1)])

X2 = np.column_stack([X2, _1, _2, _3, _4, _5])

#using min max scaling because standardization led to divide by zeor error, still had to add epsilon
epsilon = 1e-8
X2 = (X2 - np.min(X2, axis=0)) / (np.max(X2, axis=0) - np.min(X2, axis=0) + epsilon)
_, p2 = X2.shape

w = np.zeros((p2, 1))
b = 0
m2 = LinearRegression(w, b, 0.2, 0.01)

X2.shape
m2.w.shape

m2.train(X2, y, 10000)
m2.plot(X2, y)
#model trains well without regularization
