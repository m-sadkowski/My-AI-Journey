"""
-----------------------------------------------------------------------------------------
source of knowledge: https://www.youtube.com/watch?v=lyIqk783YSY&ab_channel=RobertSikora
-----------------------------------------------------------------------------------------
"""

import math
import numpy

"""
A perceptron is the simplest type of artificial neural network and the building block of more complex models. It is a single-layer neural network used for binary CLASSIFICATION tasks (example: classifying emails as "spam" or "not spam").
It takes multiple inputs, applies weights and a bias, sums them up, and passes the result through an activation function to produce an output (usually 0 or 1).
It learns to classify data by adjusting its weights and bias during training to minimize errors.
It can only classify data that is linearly separable - it means you can draw a straight line (in 2D), a plane (in 3D), or a hyperplane (in higher dimensions) that perfectly separates the data points into two classes.
It laid the groundwork for more advanced neural networks like multi-layer perceptrons (MLPs).
"""
class Perceptron:
    def __init__(self):
        """
        The activation function decides whether and how strongly a neuron should be "activated" based on the weighted sum of inputs and bias.
        Without an activation function, a neural network would simply be a linear regression model. Activation functions introduce non-linearity, enabling the network to learn complex patterns.
        """
        self.activation_function = lambda x: 1.5 * math.tanh(x) + 1 # Expected types of iris' are 0, 1, 2 - tanh can achieve values from -1 to 1 so we change the function into more fitting one (bigger set of values, offset)
        """
        Weights are parameters that determine how strongly each input influences the output. Each input in a neural network is multiplied by its corresponding weight.
        Weights are crucial in machine learning because the model learns to adjust these values to minimize prediction errors.
        """
        self.weights = []
        """
        Bias is an additional parameter in a model that allows the output of the activation function to be shifted. It acts as an "offset" or "threshold" that helps the model better fit the data.
        Without bias, the model might struggle to fit data, especially when the data does not pass through the origin.
        """
        self.bias = 0

    def forward(self, x: list[float]) -> float:
        """
        Output = Activation Function(∑(Input×Weight)+Bias)
        """
        return self.activation_function(numpy.dot(x, self.weights) + self.bias)

    def train(self, X_train: list[list[float]], y_expected: list[float], n_iter: int, learning_rate: float):
        number_of_inputs = len(X_train[0])
        self.weights = numpy.random.randn(number_of_inputs) # We start with random values for better results
        self.bias = numpy.random.randn()
        for _ in range(n_iter):
            for i, x in enumerate(X_train):
                y_predicted = self.forward(x)
                error = y_expected[i] - y_predicted
                correction = error * learning_rate
                self.weights = self.weights + correction * x
                self.bias = self.bias + correction

    def predict(self, X: list[list[float]]) -> list[float]:
        predictions = []
        for _, x in enumerate(X):
            output = self.forward(x)
            predictions.append(output)

        return predictions


