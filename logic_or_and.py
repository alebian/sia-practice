import numpy as np
from simple_perceptron import SimplePerceptron

coeff = 0.05

training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
number_of_training_inputs = training_inputs.shape[0]
desired_output = np.array([[-1], [1], [1], [1]])

print('Training logic OR:')
or_net = SimplePerceptron(2, 1, coeff)
out = or_net.fit(training_inputs, desired_output)
print('[+] Training complete. Weights:')
print(or_net._weights)
print('[+] Trained output:')
print(out)
print('[+] Prediction for [0, 0]')
print(or_net.predict(training_inputs[0]))

training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
number_of_training_inputs = training_inputs.shape[0]
desired_output = np.array([[-1], [-1], [-1], [1]])

print('Training logic AND:')
and_net = SimplePerceptron(2, 1, coeff)
out = and_net.fit(training_inputs, desired_output)
print('[+] Training complete. Weights:')
print(and_net._weights)
print('[+] Trained output:')
print(out)
print('[+] Prediction for [0, 0]')
print(or_net.predict(training_inputs[0]))
