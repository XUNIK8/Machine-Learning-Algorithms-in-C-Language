#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Use banknote.data with 4 features and 2 labels(binary)  
#define N_FEATURES 4
#define MAX_LENGTH 2000
#define INF 1000000000


typedef struct BinaryRegressionTree
{
    struct BinaryRegressionTree* left;
    struct BinaryRegressionTree* right;
    float value;
    int split_label;
    float split_threshold;
}BinaryRegressionTree;


BinaryRegressionTree* createBinaryRegressionTree() {
    BinaryRegressionTree *tree = (BinaryRegressionTree*)malloc(sizeof(BinaryRegressionTree));
    tree->left = NULL;
    tree->right = NULL;
    tree->value = 0.0;
    tree->split_label = -1;
    tree->split_threshold = 0.0;
    return tree;
}


void destroyTree(BinaryRegressionTree* root) {
    if (root == NULL) {
        return;
    }
    destroyTree(root->left);
    destroyTree(root->right);
    free(root);
}


int isLeave(BinaryRegressionTree* tree) {
    return (tree->left == NULL) || (tree->right == NULL);
}

BinaryRegressionTree* train_regression_tree(float** train_x, float* train_y, int n, int max_depth);

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
    int i, j, length = 0, num;
    float x[MAX_LENGTH][N_FEATURES], x_shuffled[MAX_LENGTH][N_FEATURES];
    float y[MAX_LENGTH], y_shuffled[MAX_LENGTH];
    int seed[MAX_LENGTH];

    // Load data from csv
    FILE* in_file = fopen("dataset/banknote.csv", "r");
    while (1) {
        for (j = 0; j < N_FEATURES; j++) {
            fscanf(in_file, "%f,", &x[length][j]);
        }
        num = fscanf(in_file, "%f\n", &y[length]);
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
    float *y_hat = (float*)malloc(sizeof(float) * split_length);

    // Copy the train dataset
    float** train_x = (float**)malloc(sizeof(float*) * (length - split_length));
    float* train_y = (float*)malloc(sizeof(float) * (length - split_length));
    for (i = 0; i < length - split_length; i++) {
        train_x[i] = (float*)malloc(sizeof(float*) * N_FEATURES);
    }
    for (i = split_length; i < length; i++) {
        for (j = 0; j < N_FEATURES; j++) {
            train_x[i - split_length][j] = x_shuffled[i][j];
        }
        train_y[i - split_length] = y_shuffled[i];
    }

    // Regression Trees
    BinaryRegressionTree* root = \
        train_regression_tree(train_x, train_y, length - split_length, 2);
    BinaryRegressionTree* p;
    for (i = 0; i < split_length; i++) {
        p = root;
        while (!isLeave(p)) {
            if (x_shuffled[i][p->split_label] <= p->split_threshold) {
                // Turn left
                p = p->left;
            }
            else {
                // Turn right
                p = p->right;
            }
        }
        y_hat[i] = p->value;
    }

    float mse = 0.0;

    for (i = 0; i < split_length; i++) {
        mse += (y_hat[i] - y_shuffled[i]) * (y_hat[i] - y_shuffled[i]);
        
    }
    mse /= (float)split_length;
 
    printf("rmse = %.2f\n", sqrt(mse));

    for (i = 0; i < length - split_length; i++) {
        free(train_x[i]);
    }
    free(train_x);
    free(train_y);
    free(y_hat);

}

/*
 * Function of training regression trees
 */
BinaryRegressionTree* train_regression_tree(float** train_x, float* train_y, int n, int max_depth) {
    BinaryRegressionTree* root = createBinaryRegressionTree();
    
    if (max_depth == 0) {
        // Get to the leaf
        float value = 0.0;
        for (int i = 0; i < n; i++) {
            value += train_y[i];
        }
        root->value = value;
        return root;
    }
    
    float split_threshold = 0;
    int split_label = -1;
    float min_gini_label = INF;

    int* d1 = (int*)malloc(sizeof(int) * n);
    int* d2 = (int*)malloc(sizeof(int) * n);
    float* gini = (float*)malloc(sizeof(float) * n);
    int i, j, k, d1_length, d2_length;
    float c1, c2;
    float s, min_s_gini = INF;
    int min_s_index;
    float* min_gini = (float*)malloc(sizeof(float) * n);
    // Go through all the features
    for (i = 0; i < N_FEATURES; i++) {
        // Go through all the node
        for (j = 0; j < n; j++) {
            s = train_y[j];
            d1_length = 0;
            d2_length = 0;
            for (k = 0; k < n; k++) {
                if (train_y[k] <= s) {
                    d1[d1_length++] = k;
                }
                else {
                    d2[d2_length++] = k;
                }
            }
            // Calculate Gini
            c1 = 0.0;
            for (k = 0; k < d1_length; k++) {
                c1 += train_y[d1[k]];
            }
            c1 = c1 / (float)d1_length;

            c2 = 0.0;
            for (k = 0; k < d2_length; k++) {
                c2 += train_y[d2[k]];
            }
            c2 = c2 / (float)d2_length;

            gini[j] = 0.0;
            for (k = 0; k < d1_length; k++) {
                gini[j] += (train_y[d1[k]] - c1) * (train_y[d1[k]] - c1);
            }
            for (k = 0; k < d2_length; k++) {
                gini[j] += (train_y[d2[k]] - c2) * (train_y[d2[k]] - c2);
            }
        }
        
        // Find min Gini
        min_s_index = -1;
        for (j = 0; j < n; j++) {
            if (gini[j] < min_s_gini) {
                min_s_gini = gini[j];
                min_s_index = j;
            }
        }
        s = train_y[min_s_index];

        // Updata label and shreshold
        if (min_s_gini < min_gini_label) {
            min_gini_label = min_s_gini;
            split_label = i;
            split_threshold = s;
        }
    }

    // Refactoring dataset
    d1_length = 0;
    d2_length = 0;
    for (k = 0; k < n; k++) {
        if (train_y[k] <= split_threshold) {
            d1[d1_length++] = k;
        }
        else {
            d2[d2_length++] = k;
        }
    }
    float** d1_train_x = (float**)malloc(sizeof(float*) * d1_length);
    float* d1_train_y = (float*)malloc(sizeof(float) * d1_length);
    float** d2_train_x = (float**)malloc(sizeof(float*) * d2_length);
    float* d2_train_y = (float*)malloc(sizeof(float) * d2_length);
    for (i = 0; i < d1_length; i++) {
        d1_train_x[i] = (float*)malloc(sizeof(float*) * N_FEATURES);
    }
    for (i = 0; i < d2_length; i++) {
        d2_train_x[i] = (float*)malloc(sizeof(float*) * N_FEATURES);
    }
    for (i = 0; i < d1_length; i++) {
        for (int j = 0; j < N_FEATURES; j++) {
            d1_train_x[i][j] = train_x[d1[i]][j];
        }
        d1_train_y[i] = train_y[d1[i]];
    }
    for (i = 0; i < d2_length; i++) {
        for (int j = 0; j < N_FEATURES; j++) {
            d2_train_x[i][j] = train_x[d2[i]][j];
        }
        d2_train_y[i] = train_y[d2[i]];
    }
    

    root->split_label = split_label;
    root->split_threshold = split_threshold;
    root->left = train_regression_tree(d1_train_x, d1_train_y, d1_length, max_depth - 1);
    root->right = train_regression_tree(d2_train_x, d2_train_y, d2_length, max_depth - 1);

    // Free storage
    for (i = 0; i < d1_length; i++) {
        free(d1_train_x[i]);
    }
    for (i = 0; i < d2_length; i++) {
        free(d2_train_x[i]);
    }
    free(d1_train_x);
    free(d2_train_x);
    free(d1_train_y);
    free(d2_train_y);
    free(d1);
    free(d2);
    free(gini);
    free(min_gini);

    return root;
}

