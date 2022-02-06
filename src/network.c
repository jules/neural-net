#include <math.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>

double* create_layer(int n_weights, int n_neurons) {
	double* x = malloc(sizeof(double) * (n_weights+1) * n_neurons);
	if (!x) {
		printf("out of memory");
		exit(1);
	}

	return x;
}

double activate(double* weights, double* inputs, int inputs_len) {
	double bias = weights[inputs_len-1];

	for (int i = 0; i < inputs_len; i++) {
		bias = bias + weights[i] * inputs[i];
	}
	return bias;
}

double transfer(double activation) {
	return 1.0 / (1.0 + exp(-activation));
}

double* forward_propagate(double* hidden_layer, int hidden_neurons, int hidden_weights, 
        double* output_layer, int output_neurons, int output_weights, 
        double* inputs, int inputs_length) {
	double* hidden_outputs = malloc(sizeof(double) * hidden_neurons);
    if (!hidden_outputs) {
        printf("out of memory");
        exit(1);
    }

	for (int i = 0; i < hidden_neurons; i++) {
	    double activation = activate(&hidden_layer[i], inputs, inputs_length);
	    hidden_outputs[i] = transfer(activation);
	}
	
	double* outputs = malloc(sizeof(double) * output_neurons);
    if (!outputs) {
        printf("out of memory");
        exit(1);
    }

	for (int i = 0; i < output_neurons; i++) {
	    double activation = activate(&output_layer[i], hidden_outputs, hidden_neurons);
	    outputs[i] = transfer(activation);
	}
	
	return outputs;
}

double transfer_derivative(double output) {
	return output * (1.0 - output);
}

void backward_propagate_error(double* hidden_layer, int hidden_neurons, int hidden_weights, 
        double* output_layer, int output_neurons, int output_weights, 
        double* outputs, int outputs_length,
        double* expected, int expected_length) {

}
