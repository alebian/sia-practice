from multilayer_feed_forward_network import MultilayerFeedForwardNetwork as Network

layers = [2, 3, 1]

network = Network(layers)

training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
number_of_training_inputs = training_inputs.shape[0]
desired_output = np.array([[-1], [1], [1], [1]])

network.fit(training_inputs[0], desired_output[0])
