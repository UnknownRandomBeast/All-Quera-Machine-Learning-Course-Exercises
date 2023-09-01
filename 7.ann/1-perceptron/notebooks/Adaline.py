import numpy as np
from sklearn.metrics import accuracy_score
class Adaline:
    def __init__(self):
        self.weights = None

    def weighting(self, input):
        return np.dot(self.weights, input)

    def activation(self, weighted_input):
        return weighted_input

    def predict(self, inputs):
        samples_count = inputs.shape[0]

        bias_added_inputs = np.insert(inputs, 0, 1, axis=1)
        
        weighted_inputs = np.zeros(samples_count)

        for i in range(samples_count):
            weighted_inputs[i] = self.weighting(bias_added_inputs[i])

        activated_inputs = self.activation(weighted_inputs)

        for i in range(samples_count):
            if activated_inputs[i] >= 0:
                activated_inputs[i] = 1
            else:
                activated_inputs[i] = -1

        return activated_inputs

    def fit(self, inputs, outputs, learning_rate=0.1, epochs=64):
        bias_added_inputs = np.insert(inputs, 0, 1, axis=1)

        self.weights = np.random.rand(bias_added_inputs.shape[1])
        
        for i in range(epochs):
            od = self.predict(inputs)
            d_w = learning_rate * (outputs - od)
            d_w = d_w.reshape((d_w.shape[0], 1))
            d_w = np.tile(d_w, (1, bias_added_inputs.shape[1]))
            d_w = d_w * bias_added_inputs
            d_w = np.sum(d_w, axis=0)

            self.weights += d_w
            output_pred = self.predict(inputs)

            print(f'model accuracy after epoch{i+1}: {accuracy_score(outputs, output_pred)}')

