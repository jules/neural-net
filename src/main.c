#include <stdio.h>
#include "network.c"

int main() {
	double* hidden_layer = create_layer(1, 2);
	double* output_layer = create_layer(2, 1);
	for (int i = 0; i < 3; i++) {
		printf("%f ", *(hidden_layer + i));
	}

	printf("\n");
	for (int i = 0; i < 2; i++) {
		printf("%f ", *(output_layer + i));
	}

	printf("\n");
	for (int i = 0; i < 2; i++) {
		printf("%f ", *(output_layer + 2 + i));
	}

	printf("\n");
    return 0;
}

