#include <math.h>
#if defined __linux__ ||  defined __unix__ ||  defined __APPLE__
# include <unistd.h>
#endif
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
    double* hidden_deltas;

    int     n_hidden_weights;
    double* hidden_weights;

    // Output layer
    int     n_output_neurons;
    double* output_neurons;
    double* output_deltas;

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
    n.hidden_deltas = create_layer(n.n_hidden_weights + 1);
    n.n_hidden_weights = n_hidden_weights;
    n.hidden_weights = malloc(sizeof(double) * (n.n_hidden_weights + 1) * n.n_hidden_neurons);
    if (!n.hidden_weights) {
        printf("out of memory");
        exit(1);
    }

    n.n_output_neurons = n_output_neurons;
    n.output_neurons = create_layer(n.n_output_neurons);
    n.output_deltas = create_layer(n.n_output_weights + 1);
    n.n_output_weights = n_output_weights;
    n.output_weights = malloc(sizeof(double*) * (n.n_output_weights + 1) * n.n_output_neurons);
    if (!n.output_weights) {
        printf("out of memory");
        exit(1);
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

// Runs a full 'forward propagation' through the whole network.
void forward_propagate(struct network* n) {
    // todo: abstractions lol
	for (int i = 0; i < n->n_hidden_neurons; i++) {
	    double activation = activate(&n->hidden_weights[i], n->inputs, n->n_inputs);
	    n->hidden_neurons[i] = transfer(activation);
	}
	
	for (int i = 0; i < n->n_output_neurons; i++) {
	    double activation = activate(&n->output_weights[i], n->hidden_neurons, n->n_hidden_neurons);
	    n->output_neurons[i] = transfer(activation);
	}
}

double transfer_derivative(double output) {
	return output * (1.0 - output);
}

void backward_propagate_error(struct network* n, double* expected) {
    double* errors = malloc(sizeof(double) * (n->n_hidden_neurons * n->n_output_neurons));
    
    // todo: can we abstract better here?
    // Set deltas and add errors for output layer
    for (int i = 0; i < n->n_output_neurons; i++) {
        errors[i] = n->output_neurons[i] - expected[i];
    }

    for (int i = 0; i < n->n_output_neurons; i++) {
        n->output_deltas[i] = errors[i] * transfer_derivative(n->output_neurons[i]);
    }

    // Set deltas and add errors for hidden layer
    for (int i = 0; i < n->n_hidden_neurons; i++) {
        double error;
        for (int j = 0; j < n->n_output_neurons; i++) {
            error = error + (n->output_weights[j * n->n_output_weights + i] * n->output_deltas[j]);
        }

        errors[i + n->n_output_neurons] = n->output_neurons[i] - expected[i];
    }

    for (int i = 0; i < n->n_hidden_neurons; i++) {
        n->hidden_deltas[i] = errors[i + n->n_output_neurons] * transfer_derivative(n->hidden_neurons[i]);
    }
}

void update_weights(struct network* n, double* row, int row_length, double l_rate) {
    // todo: abstract into functions
    for (int i = 0; i < n->n_hidden_neurons; i++) {
        for (int j = 0; j < row_length - 1; j++) {
            n->hidden_weights[i * n->n_hidden_weights + j] = n->hidden_weights[i * n->n_hidden_weights + j] - l_rate * n->hidden_deltas[i] * row[j];
        }

        n->hidden_weights[n->n_hidden_weights - 1] = n->hidden_weights[n->n_hidden_weights - 1] - l_rate * n->hidden_deltas[n->n_hidden_weights - 1];
    }

    for (int i = 0; i < n->n_output_neurons; i++) {
        for (int j = 0; j < n->n_hidden_neurons; j++) {
            n->output_weights[i * n->n_output_weights + j] = n->output_weights[i * n->n_output_weights + j] - l_rate * n->output_deltas[i] * n->hidden_neurons[j];
        }

        n->output_weights[n->n_output_weights - 1] = n->output_weights[n->n_output_weights - 1] - l_rate * n->output_deltas[n->n_output_weights - 1];
    }
}
