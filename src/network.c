#include <math.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>

// A dynamically sized 2-layer neural network.
struct network {
    // Input layer
    int     n_inputs;
    double* inputs;

    // Hidden layer
    int     n_hidden_neurons;
    double* hidden_neurons;
    int     n_hidden_weights;
    double* hidden_weights;

    // Output layer
    int     n_output_neurons;
    double* output_neurons;
    int     n_output_weights;
    double* output_weights;
};

double* create_layer(int n_items) {
	double* x = malloc(sizeof(double) * (n_items));
	if (!x) {
		printf("out of memory");
		exit(1);
	}

	return x;
}

struct network create_network(int n_hidden_weights, int n_hidden_neurons, 
        int n_output_weights, int n_output_neurons) {
    struct network n;
    // Amount of inputs is the same as amount of hidden neurons
    n.n_inputs = n_hidden_neurons;
    n.inputs = create_layer(n.n_inputs);

    n.n_hidden_neurons = n_hidden_neurons;
    n.hidden_neurons = create_layer(n.n_hidden_neurons);
    n.n_hidden_weights = n_hidden_weights;
    n.hidden_weights = malloc(sizeof(double*) * n.n_hidden_neurons);
    if (!n.hidden_weights) {
        printf("out of memory");
        exit(1);
    }
    for (int i = 0; i < n.n_hidden_neurons; i++) {
        n.hidden_weights[i] = create_layer(n.n_hidden_weights + 1);
    }

    n.n_output_neurons = n_output_neurons;
    n.output_neurons = create_layer(n.n_output_neurons);
    n.n_output_weights = n_output_weights;
    n.output_weights = malloc(sizeof(double*) * n.n_output_neurons);
    if (!n.output_weights) {
        printf("out of memory");
        exit(1);
    }
    for (int i = 0; i < n.n_output_neurons; i++) {
        n.output_weights[i] = create_layer(n.n_output_weights + 1);
    }

    return n;
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

double* forward_propagate(struct network* n) {
	for (int i = 0; i < hidden_neurons; i++) {
	    double activation = activate(&n.hidden_weights, inputs, inputs_length);
	    n.hidden_neurons[i] = transfer(activation);
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

// void backward_propagate_error(double* hidden_layer, int hidden_neurons, int hidden_weights, 
//         double* output_layer, int output_neurons, int output_weights, 
//         double* outputs, int outputs_length,
//         double* expected, int expected_length) {
//     double* errors = malloc(sizeof(double) * 
// 
// }
