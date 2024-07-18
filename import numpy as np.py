import numpy as np 
np.random.seed(50) 
input_size = 2 
hidden_size = 2 
output_size = 1 
W_input_hidden = np.random.randn(input_size, hidden_size) 
b_hidden = np.random.randn(1, hidden_size) 
W_hidden_output = np.random.randn(hidden_size, output_size) 
b_output = np.random.randn(1, output_size) 
print("Weights connecting input to hidden layer:") 
print(W_input_hidden) 
print("\nBiases for the hidden layer:") 
print(b_hidden) 
print("\nWeights connecting hidden to output layer:") 
print(W_hidden_output) 
print("\nBias for the output layer:") 
print(b_output)
