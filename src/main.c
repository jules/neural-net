#include <stdio.h>
#include "network.c"

void test_training() {
    // Contrived dataset to train against
    double* dataset[10];
    double data_0[3] = {2.7810836,2.550537003,0.0};
    dataset[0] = &data_0;
    double data_1[3] = {1.465489372,2.362125076,0.0};
    dataset[1] = &data_1;
    double data_2[3] = {3.396561688,4.400293529,0.0};
    dataset[2] = &data_2;
    double data_3[3] = {1.38807019,1.850220317,0.0};
    dataset[3] = &data_3;
    double data_4[3] = {3.06407232,3.005305973,0.0};
    dataset[4] = &data_4;
    double data_5[3] = {7.627531214,2.759262235,1.0};
    dataset[5] = &data_5;
    double data_6[3] = {5.332441248,2.088626775,1.0};
    dataset[6] = &data_6;
    double data_7[3] = {6.922596716,1.77106367,1.0};
    dataset[7] = &data_7;
    double data_8[3] = {8.675418651,-0.242068655,1.0};
    dataset[8] = &data_8;
    double data_9[3] = {7.673756466,3.508563011,1.0};
    dataset[9] = &data_9;

    struct network n = create_network(2, 2, 2, 2);
    train_network(&n, &dataset, 10, 3, 0.5, 20, 2);

    printf("HIDDEN LAYER\n");
    for (int i = 0; i < n.n_hidden_neurons; i++) {
        printf("%f\n", n.hidden_neurons[i]);
        for (int j = 0; j < n.n_hidden_weights; j++) {
            printf("\t%f\n", n.hidden_weights[i][j]);
        }
    }

    printf("OUTPUT LAYER\n");
    for (int i = 0; i < n.n_output_neurons; i++) {
        printf("%f\n", n.output_neurons[i]);
        for (int j = 0; j < n.n_output_weights; j++) {
            printf("\t%f\n", n.output_weights[i][j]);
        }
    }
}

void test_backprop() {
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
}

int main() {
    test_training();
    return 0;
}
