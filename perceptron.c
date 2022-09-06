#include<stdio.h>
#include<stdlib.h>
#include<time.h>

// Use self-made data(5 for training and 4 for testing) with 5 features and 2 labels(-1 or 1)
#define INPUT_VECTOR_LENGTH 5
#define INSTANCES_LENGTH 4
#define TEST_LENGTH 4

struct instance{
	float input_vector[INPUT_VECTOR_LENGTH];	
	int label;
};
struct test{
	float input_vector[INPUT_VECTOR_LENGTH];
	int label;
};


void Perceptron(struct instance* instances,struct test* test, int epoch, float learning_rate)
{
	int i = 0;
	int j = 0;
	int outcome;
	float weight[INPUT_VECTOR_LENGTH] = {0}, bias = 0;
	
	srand(time(NULL));
	for(i = 0; i < INPUT_VECTOR_LENGTH; i++){
		weight[i] = (rand() % 10)*0.1;
	}
	
	int count = 0, index = 0;
	float prediction = 0;
	
	// Start PCN train
	while(count < epoch){
		prediction = 0;

		for (i = 0; i < INPUT_VECTOR_LENGTH; i ++)
			prediction += instances[index].input_vector[i]*weight[i];
		prediction += bias;

		if(prediction * instances[index].label <= 0){
			for (i = 0; i < INPUT_VECTOR_LENGTH; i ++)
				weight[i] += learning_rate*instances[index].label*instances[index].input_vector[i];
			
			bias += learning_rate*instances[index].label;
		}
		
		printf("%d    weight: ", count);
		for(i = 0; i < INPUT_VECTOR_LENGTH; i ++){
			printf("%f ", weight[i]);
		}
		
		// Print the outcome of training(including the updated weights and the comparison of labels)
		printf(" bias: %f", bias);
		printf("instances: %d label: %d prediction: %f\n", index, instances[index].label, prediction);
		
		count ++;
		index ++;
		if(index == INSTANCES_LENGTH)
			index = 0;		
	}
	
	// Start PCN test
	for(j;j < TEST_LENGTH;j++){
		prediction = 0;
		for (i = 0; i < INPUT_VECTOR_LENGTH; i ++){
			
			prediction += test[j].input_vector[i]*weight[i];
		}
		prediction += bias;
		if (prediction<0)
			outcome = -1;
		else
			outcome = 1;

		// Print the outcome of testing(including real label and predicted label)
		printf("test[%d] : label =  %d ; prediction = %d\n",j,test[j].label,outcome);
	}	
}

int main()
{	
	int i;
	float weight[INPUT_VECTOR_LENGTH] = {0}, bias = 0;
	int epoch = 100;
	float learning_rate = 0.01;
	// Train dataset(5 features with 1 label)
	struct instance instances[INSTANCES_LENGTH] = 
	{
		{{1,1,1,1,1},1},
		{{-1,-1,-1,-1,-1},-1},
		{{1,-1,1,1,-1},1},
		{{-1,1,-1,-1,1},-1}
	};
	// Test dataset
	struct test test[INSTANCES_LENGTH] = 
	{
		{{1,1,1,1,1},1},
		{{-1,-1,-1,-1,-1},-1},
		{{1,-1,1,1,-1},1},
		{{-1,-1,-1,-1,-1},-1}
	};
	Perceptron(instances, test,epoch, learning_rate);
	
	return 0;
} 
