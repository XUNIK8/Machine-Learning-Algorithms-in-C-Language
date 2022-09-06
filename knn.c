#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Use abalone.data with 8 features and 3 labels
#define N_FEATURES 8
#define N_LABELS 3
#define MAX_LENGTH 5000
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

/*
 * KNN Algorithm 
 */
int main() {
    // K-Fold 
    int k = 5;
    int i, j, t, min_index, label, max_freq, hit_num, length = 0, num;
    float min_dist, diff;
    float x[MAX_LENGTH][N_FEATURES], x_shuffled[MAX_LENGTH][N_FEATURES];
    int y[MAX_LENGTH], y_shuffled[MAX_LENGTH];
    int labels_freq[N_LABELS];
    int seed[MAX_LENGTH];

    // Load data 
    FILE* in_file = fopen("dataset/abalone.csv", "r");
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
    
    // y_hat is the predicticted outcome(label) 
    int *y_hat = (int*)malloc(sizeof(int) * split_length);
    // i = [split_length, length] is train dataset; [0, split_length - 1] is test dataset
    float *distance = (float*)malloc(sizeof(float) * (length-split_length));
    int *min_samples = (int*)malloc(sizeof(int) * k);
    for (i = 0; i < length-split_length; i++) {
        distance[i] = 0.0;
    }

    // Start KNN 
    for (i = 0; i < split_length; i++) {
        // calculate distances
        for (j = split_length; j < length; j++) {
            for (t = 0; t < N_FEATURES; t++) {
                diff = (x_shuffled[i][t] - x_shuffled[j][t]);
                distance[j - split_length] = diff * diff;
            }
        }
        // Find K min_samples
        for (t = 0; t < k; t++) {
            min_dist = INF;
            min_index = -1;
            for (j = 0; j < length - split_length; j++) {
                if (distance[j] < min_dist) {
                    min_dist = distance[j];
                    min_index = j;
                }
            }
            min_samples[t] = min_index;
            distance[min_index] = INF;
        }
        // Majority method 
        for (t = 0; t < k; t++) {
            min_samples[t] += split_length;
        }
        for (t = 0; t < N_LABELS; t++) {
            labels_freq[t] = 0;
        }
        // Calculate ratio of each predicted label
        for (t = 0; t < k; t++) {
            label = y_shuffled[min_samples[t]];
            labels_freq[label]++;
        }
        label = 0;
        max_freq = labels_freq[0];
        for (t = 1; t < N_LABELS; t++) {
            if (labels_freq[t] > max_freq) {
                max_freq = labels_freq[t];
                label = t;
            }
        }
        // Get the final predicted label 
        y_hat[i] = label;
    }

    // Calculate accuracy 
    hit_num = 0;
    for (i = 0; i < split_length; i++) {
        if (y_hat[i] == y_shuffled[i]) {
            hit_num++;
        }
    }
    printf("Final accuracy is %.2f\n", (float)hit_num / split_length);


    free(distance);
    free(min_samples);
    free(y_hat);
}





