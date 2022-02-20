#include <stdio.h>
#include "network.c"

int main() {
    struct network n = create_network(3, 1, 2, 2);

    n.hidden_neurons[0] = 0.7105668883115941;
    n.hidden_weights[0][0] = 0.13436424411240122;
    n.hidden_weights[0][1] = 0.8474337369372327;
    n.hidden_weights[0][2] = 0.763774618976614;

    n.output_neurons[0] = 0.6213859615555266;
    n.output_weights[0][0] = 0.2550690257394217;
    n.output_weights[0][1] = 0.49543508709194095;

    n.output_neurons[1] = 0.6573693455986976;
    n.output_weights[1][0] = 0.4494910647887381;
    n.output_weights[1][1] = 0.651592972722763;

    double expected[2] = {0.0, 1.0};
    backward_propagate_error(&n, &expected);
    
    for (int i = 0; i < n.n_hidden_neurons; i++) {
        printf("%f\n", n.hidden_deltas[i]);
    }

    for (int i = 0; i < n.n_output_neurons; i++) {
        printf("%f\n", n.output_deltas[i]);
    }

    return 0;
}

