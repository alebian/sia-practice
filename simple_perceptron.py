import numpy as np

class SimplePerceptron:
    def __init__(self, input_layer_size, output_layer_size, coeff = 0.05):
        self._bias = -1
        self._coeff = coeff
        self._input_layer_size = input_layer_size
        self._output_layer_size = output_layer_size
        # Weights is a matrix in which each column represents the weights from every input
        # to each output and has the bias in each column
        self._weights = np.random.rand(input_layer_size + 1, output_layer_size) - 0.5
        self._weights[0:(output_layer_size - 1)] = self._bias
        self._activation_function = lambda x: np.sign(x)
        self._activation_function_derivative = lambda x: 1

    def fit(self, inputs, desired_output, iterations = 10000):
        fit_inputs_size = inputs.shape[0]
        inputs_with_bias = np.column_stack([(np.zeros(fit_inputs_size) + self._bias), inputs])
        out = np.zeros((fit_inputs_size, self._output_layer_size))

        for _ in range(iterations):
            # Update weights for each output
            H = np.dot(inputs_with_bias, self._weights)
            out = np.array(list(map(self._activation_function, H)))
            delta = desired_output - out

            delta_w = lambda i, j, k: self._coeff * delta[i][j] * inputs_with_bias[i][k] * self._activation_function_derivative(H[i][j])

            for input_, delta_, h in zip(inputs_with_bias, delta, H):
                for j in range(self._output_layer_size):
                    for k in range(self._input_layer_size + 1):
                        self._weights[k][j] = self._weights[k][j] + self._coeff * delta_[j] * input_[k] * self._activation_function_derivative(h[j])
            # for i in range(fit_inputs_size):
            #     for j in range(self._output_layer_size):
            #         for k in range(self._input_layer_size + 1):
            #             self._weights[k][j] = self._weights[k][j] + delta_w(i, j, k)

        return out

    def predict(self, input_array):
        input_with_bias = np.concatenate(([self._bias], input_array))
        out = np.zeros(self._output_layer_size)
        for i in range(self._output_layer_size):
            h = np.dot(input_with_bias, self._weights[:,i])
            out[i] = self._activation_function(h)
        return out
