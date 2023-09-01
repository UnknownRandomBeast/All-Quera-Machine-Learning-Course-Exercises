import numpy as np
from sklearn.metrics import accuracy_score
class Perceptron:
    def __init__(self):
        self.weights = None

    def weighting(self, input):
        return np.dot(self.weights, input)

    def activation(self, weighted_input):
        if weighted_input >= 0:
            return 1
        else:
            return -1

    def predict(self, inputs):
        bias_added_inputs = np.insert(inputs, 0, 1, axis=1)

        outputs = np.zeros(inputs.shape[0])

        for i in range(bias_added_inputs.shape[0]):
            weighted_input = self.weighting(bias_added_inputs[i])
            activation_output = self.activation(weighted_input)
            outputs[i] = activation_output

        return outputs

    def fit(self, inputs, outputs, learning_rate, epochs):
        bias_added_inputs = np.insert(inputs, 0, 1, axis=1)
        
        self.weights = np.random.rand(5)

        for i in range(epochs):
            for j in range(bias_added_inputs.shape[0]):
                weighted_input = self.weighting(bias_added_inputs[j])
                activation_output = self.activation(weighted_input)
                error = outputs[j] - activation_output
                self.weights += learning_rate * error * bias_added_inputs[j]

            ouput_pred = self.predict(inputs)
            print(f'model accuracy after epoch{i+1}: {accuracy_score(outputs, ouput_pred)}')

