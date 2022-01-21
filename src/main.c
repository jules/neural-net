#include <stdio.h>
#include <stdlib.h>
#include "network.h"

int main() {
	int *hidden_layer = create_layer(1, 2);
	int *output_layer = create_layer(2, 1);
	for (int i = 0; i < 3; i++) {
		printf("%d ", *(hidden_layer + i));
	}

	printf("\n");
	for (int i = 0; i < 2; i++) {
		printf("%d ", *(output_layer + i));
	}

	printf("\n");
	for (int i = 0; i < 2; i++) {
		printf("%d ", *(output_layer + 2 + i));
	}

	printf("\n");
    return 0;
}

double activate(int *weights, int *inputs, int inputs_len) {
	double bias = *(weights + inputs_len);

	for (int i = 0; i < inputs_len; i++) {
		bias = bias + (*(weights + i) * *(inputs + i));
	}
	return bias;
}

double transfer(double activation) {
    return 1.0 / (1.0 + exp(-activation));
}

int* forward_propagate(int* hidden_layer, int hidden_neurons, int hidden_weights, int* output_layer, int output_neurons, int output_weights, int* inputs, int inputs_length) {
    double hidden_outputs[hidden_neurons];
    for (int i = 0; i < hidden_neurons; i++) {
        double activation = activate(hidden_layer[i], inputs, inputs_length);
        hidden_outputs[i] = transfer(activation);
    }

    double outputs[output_neurons];
    for (int i = 0; i < output_neurons; i++) {
        double activation = activate(output_layer[i], *hidden_outputs, hidden_neurons);
        outputs[i] = transfer(activation);
    }

    return *outputs;
}