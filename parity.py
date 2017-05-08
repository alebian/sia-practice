import numpy as np
from simple_perceptron import SimplePerceptron

training_inputs = np.array([[0, 0, 0], [0, 1, 1], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]])
number_of_training_inputs = training_inputs.shape[0]
desired_output = np.array([[-1, 0], [-1, 0], [1, 0], [1, 0], [-1, 0], [1, 0]])

print('Training parity:')
parity_net = SimplePerceptron(3, 2)
out = parity_net.fit(training_inputs, desired_output)
print('[+] Training complete. Weights:')
print(parity_net._weights)
print('[+] Trained output:')
print(out)

print('Now testing:')
test_input_1 = np.array([1, 0, 1])
print('For')
print(test_input_1)
print('Expected:')
print([-1])
print('Got:')
print(parity_net.predict(test_input_1))

test_input_2 = np.array([0, 1, 0])
print('For')
print(test_input_2)
print('Expected:')
print([1])
print('Got:')
print(parity_net.predict(test_input_2))
