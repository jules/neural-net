int* create_layer(int n_weights, int n_neurons);
int* forward_propagate(int* hidden_layer, int hidden_neurons, int hidden_weights, int* output_layer, int output_neurons, int output_weights, int* inputs, int inputs_length);
double activate(int *weights, int *inputs, int inputs_len);
double transfer(double activation);
