import numpy as np
from math import exp

class MultilayerFeedForwardNetwork:
    def __init__(self, architecture, coeff = 0.05):
        """
        Initialize a feed forward multi layered neural network
        :param architecture: array containing the description of each layer
        """
        self._architecture = architecture
        self._bias = -1
        self._coeff = coeff
        self._weights = []
        self._activation_function = lambda x: np.tanh(x)
        self._activation_function_derivative = lambda x: 4 / (exp(-x) + exp(x))**2
        for i in range(len(architecture) - 1):
            # Add the bias to each
            self._weights.append(np.random.rand(architecture[i] + 1, architecture[i + 1]))
            self._weights[i][0,:] = self._bias

    def fit(self, input_array, desired_output):
        prediction, intermediate_inputs = self._prediction_with_steps(input_array)
        error = desired_output - prediction

        delta_w = lambda error, h, inp: self._coeff * error * self._activation_function_derivative(h) * inp

        for i in range(len(self._weights) - 1, 0, -1):
            W = self._weights[i]
            I = intermediate_inputs[i]
            h = np.dot(I, W)
            for j in range(len(error)):
                for k in range(len(W)):
                    W[k][j] = W[k][j] + delta_w(error[j], h[j], I[k])
            error = []

    def _add_bias(self, array):
        return np.concatenate(([self._bias], array))

    def _prediction_with_steps(self, input_array):
        inputs = []
        input_with_bias = self._add_bias(input_array)
        inputs.append(input_with_bias)

        for i in range(len(self._weights)):
            new_input = np.dot(input_with_bias, self._weights[i])
            new_input = list(map(self._activation_function, new_input.tolist()))
            input_with_bias = self._add_bias(new_input)
            inputs.append(input_with_bias)

        return (input_with_bias[1:], inputs)

    def predict(self, input_array):
        return self._prediction_with_steps(input_array)[0]
