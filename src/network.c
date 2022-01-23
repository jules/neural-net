#include <math.h>

int* create_layer(int n_weights, int n_neurons) {
	double layer[n_weights+1][n_neurons];
	return layer;
}

double activate(int *weights, double inputs[], int inputs_len) {
	double bias = *(weights + inputs_len);

	for (int i = 0; i < inputs_len; i++) {
		bias = bias + (*(weights + i) * *(inputs + i));
	}
	return bias;
}

double transfer(double activation) {
    return 1.0 / (1.0 + exp(-activation));
}

double* forward_propagate(int* hidden_layer, int hidden_neurons, int hidden_weights, int* output_layer, int output_neurons, int output_weights, int* inputs, int inputs_length) {
    double hidden_outputs[hidden_neurons];
    for (int i = 0; i < hidden_neurons; i++) {
        double activation = activate(&hidden_layer[i], inputs, inputs_length);
        hidden_outputs[i] = transfer(activation);
    }

    double outputs[output_neurons];
    for (int i = 0; i < output_neurons; i++) {
        double activation = activate(&output_layer[i], hidden_outputs, hidden_neurons);
        outputs[i] = transfer(activation);
    }

    return outputs;
}
