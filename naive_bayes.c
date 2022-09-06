#include <stdio.h>
#include <stdlib.h>

// Use iris.data with 4 features and 3 labels
#define N_FEATURES 4
#define N_LABELS 3
#define MAX_LENGTH 1000
#define INF 1000000000


// shuffle
void random_shuffle(int * array, int len)
{
    int * p = array, temp, pos;
    for (int i = 1; i < len; ++i)
    {
        pos  = rand() % i;
        temp = *p;
        *p++ = array[pos];
        array[pos] = temp;
    }
}


int main() {
    int i, j, t, length = 0, num, hit_num;
    float x[MAX_LENGTH][N_FEATURES], x_shuffled[MAX_LENGTH][N_FEATURES];
    int y[MAX_LENGTH], y_shuffled[MAX_LENGTH];
    int seed[MAX_LENGTH];

    // Load file from csv 
    FILE* in_file = fopen("dataset/iris.csv", "r");
    while (1) {
        for (j = 0; j < N_FEATURES; j++) {
            fscanf(in_file, "%f,", &x[length][j]);
        }
        num = fscanf(in_file, "%d\n", &y[length]);
        if (num != 1) {
            break;
        }
        length++;
    }
    fclose(in_file);

    // K-fold divide data
    for (i = 0; i < length; i++) {
        seed[i] = i;
    }
    random_shuffle(seed, length);
    for (i = 0; i < length; i++) {
        for (j = 0; j < N_FEATURES; j++) {
            x_shuffled[i][j] = x[seed[i]][j];
        }
        y_shuffled[i] = y[seed[i]];
    }

    int n_fold = 5;
    float ratio = 1 / (float)n_fold;
    int split_length = (int)(ratio * length);

    // y_hat is the predicted label
    int *y_hat = (int*)malloc(sizeof(int) * split_length);

    // Start NB(simplified)
    float *px_y = (float*)malloc(sizeof(float) * N_FEATURES);
    float py;
    float hat_value;
    float max_value = -INF;
    int label;
    int max_label;
    int x_counter, y_counter;

    for (i = 0; i < split_length; i++) {
        max_value = -INF;
        max_label = -1;
        // Find the labels with max probability
        for (label = 0; label < N_LABELS; label++) {
            x_counter = 0;
            y_counter = 0;
            // Calculate the ratio of each label
            for (j = split_length; j < length; j++) {
                if (y_shuffled[j] == label) {
                    y_counter++;
                    if (x_shuffled[j][label] == x_shuffled[i][label]) {
                        x_counter++;
                    }
                }
            }
            py = (float)y_counter / (length - split_length);
            if (x_counter == 0) {
                px_y[label] = (x_counter + 1) / (float)(y_counter + 1);
            }
            else {
                px_y[label] = x_counter / (float)y_counter;
            }

            // Find the labels with max probability
            hat_value = 1.0;
            for (j = 0; j < N_FEATURES; j++) {
                hat_value *= px_y[j];
            }
            hat_value *= py;

            if (hat_value > max_value) {
                max_value = hat_value;
                max_label = label;
            }
        }
        y_hat[i] = max_label;
    }

    // Calculate accuracy
    hit_num = 0;
    for (i = 0; i < split_length; i++) {
        if (y_hat[i] == y_shuffled[i]) {
            hit_num++;
        }
    }
    printf("Final accuracy is %.2f\n", (float)hit_num / split_length);

    free(px_y);
    free(y_hat);
}

