from multilayer_feed_forward_network import MultilayerFeedForwardNetwork as Network

layers = [
    2, # Input layer, 3 neurons
    3, # First hidden layer, 5 neurons
    2, # Second hidden layer, 4 neurons
    # 7, # Third hidden layer, 7 neurons
    # 2 # Output layer, 2 neurons
]

network = Network(layers)

input_1 = [1, 2]

result = network.predict(input_1)

print(result)

network.fit(input_1, [1])
