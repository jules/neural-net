#include <stdio.h>
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

