#include <stdio.h>
#include <stdlib.h>

// Use insurance.data 
#define S 1000

// y=ax + b

int main() {
    // Initialize attributes
    int i;
    int length = 0;
    float x[S], y[S];
    float sumX = 0, sumX2 = 0, sumY = 0, sumXY = 0;
    float a, b;

    // Load data from csv
    FILE* in_file = fopen("dataset/insurance.csv", "r");
    if (in_file == NULL) {
        printf("Error! Could not open file\n");
        exit(-1);
    }
    while (fscanf(in_file, "%f,%f\n", &x[length], &y[length]) == 2) {
        length++;
    }
    fclose(in_file);
    
    // Calculate x and y and x^2 and sum of xy 
    for (i = 1; i < length; i++) {
        sumX += x[i];
        sumX2 += x[i] * x[i];
        sumY += y[i];
        sumXY += x[i] * y[i];
    }

    // Calculate a and b 
    a = (length*sumXY - sumX*sumY) / (length*sumX2 - sumX*sumX);
    b = (sumY - a*sumX) / length;

    printf("Values are: a = %0.2f and b = %0.2f\n", a, b);
    printf("Equation of best fit is : y = %0.2fx + %0.2f\n", a, b);
    return 0;

}

