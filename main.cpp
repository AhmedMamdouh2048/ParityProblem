#include "DATASET.h"
#include "InputOutput.h"
#include "NeuralNetwork.h"
int main()
{
	srand(time(NULL));
	string ErrorType = "SquareErr";
	layer *layers = new layer[3];
	layers[0].put(12, "");
	layers[1].put(32, "satLinear"); //54//60(.09,2048)//32(.095,2048)//
	layers[2].put(1, "satlinear2");

	int i = 0;
	string s = "Out";
	while (2048)
	{
		NeuralNetwork NN(layers, 3);
		matrix<float> X(12, 4096);
		matrix<float> Y(1, 4096);
		InputOutput(X, Y, ErrorType);
        clock_t start = clock();
		NN.train(X, Y,.095, 500, 2048, "Adam", 10, ErrorType,0,0);
        s.append("1");
        NN.test(X,Y,s,0);
        clock_t end = clock();
        i++;
        double duration_sec = double(end - start) / CLOCKS_PER_SEC;
		cout << "Time = " << duration_sec << endl;
		cout << "<===========================================>X<===========================================>" << endl << endl;
	}
	_getche();
	return 0;
}

