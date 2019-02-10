#include "NeuralNetwork.h"
#include "Activations.h"
#include <fstream>
#include "CharGenerate.h"


typedef Dictionary < string,matrix<float> > dictionary;
typedef matrix<float> Matrix;

//////////////////////////////////////////////////////////*********************************\////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////|*****PUBLIC MEMBER FUNCTIONS*****|////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////\*********************************/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
NeuralNetwork::NeuralNetwork(layer* mylayers,int L)
{
    numOfLayers=L;
    layers=mylayers;
    cache.setName("cache");
    parameters.setName("parameters");
    grades.setName("grades");                                              // Dictionary holding gradients of weights and biases
    maxErr=2;
    Matrix** Mw = new Matrix*[numOfLayers - 1];
    Matrix** Mb = new Matrix*[numOfLayers - 1];
    for (int i = 0; i<numOfLayers - 1; i++)                                 // L-1 = number of hidden layers + output layer
    {
        Mw[i] = new Matrix(layers[i + 1].neurons, layers[i].neurons, Random); // Mw[0] holds W1 and so on
        *Mw[i] = *Mw[i] / float(RAND_MAX);
        /*To make the standard deviation of weights = 1 and mean = 0*/
        if(Mw[i]->Rows() != 1 || Mw[i]->Columns() != 1)         // Don't calculate if dimensions are 1x1
        {
            float Wmean = Mw[i]->sumall() / (Mw[i]->Rows() * Mw[i]->Columns());
            *Mw[i] = *Mw[i] - Wmean;
            float Wstd = sqrt((Mw[i]->square()).sumall() / (Mw[i]->Rows() * Mw[i]->Columns()));
            *Mw[i] = *Mw[i] / Wstd;
        }

        if (layers[i+1].activation == "sigmoid")
            *Mw[i]=*Mw[i] * sqrt(2/layers[i].neurons);
        else if (layers[i+1].activation == "mytanh")
            *Mw[i]=*Mw[i] * sqrt(1/layers[i].neurons);
        else if (layers[i+1].activation == "relu")
            *Mw[i]=*Mw[i] * sqrt(2/layers[i].neurons);
        else if (layers[i+1].activation == "leakyRelu")
            *Mw[i]=*Mw[i] * sqrt(2/layers[i].neurons);
        else if (layers[i+1].activation == "satLinear")
            *Mw[i]=*Mw[i] * sqrt(1/layers[i].neurons);
        else if (layers[i+1].activation == "Linear")
            *Mw[i]=*Mw[i] * sqrt(2/layers[i].neurons);
        else if (layers[i+1].activation == "satlinear2")
            {*Mw[i]=*Mw[i] * sqrt(1/layers[i].neurons);}
        else if (layers[i+1].activation == "satlinear3")
            {*Mw[i]=*Mw[i] * sqrt(1/layers[i].neurons);}

        parameters.put(CharGen("W", i + 1), *Mw[i]);

        Mb[i] = new Matrix(layers[i + 1].neurons, 1, Random);
        *Mb[i] = *Mb[i] / float(RAND_MAX);
        /*To make the standard deviation of biases = 1 and mean = 0*/
        if(Mb[i]->Rows() != 1 || Mb[i]->Columns() != 1)         // Don't calculate if dimensions are 1x1
        {
            float bmean = Mb[i]->sumall() / (Mb[i]->Rows() * Mb[i]->Columns());
            *Mb[i] = *Mb[i] - bmean;
            float bstd = sqrt((Mb[i]->square()).sumall() / (Mb[i]->Rows() * Mb[i]->Columns()));
            *Mb[i] = *Mb[i] / bstd;
        }

        if (layers[i+1].activation == "sigmoid")
            *Mb[i]=*Mb[i] * sqrt(2/layers[i].neurons);
        else if (layers[i+1].activation == "mytanh")
            *Mb[i]=*Mb[i] * sqrt(1/layers[i].neurons);
        else if (layers[i+1].activation == "relu")
            *Mb[i]=*Mb[i] * sqrt(2/layers[i].neurons);
        else if (layers[i+1].activation == "leakyRelu")
            *Mb[i]=*Mb[i] * sqrt(2/layers[i].neurons);
        else if (layers[i+1].activation == "satLinear")
            *Mb[i]=*Mb[i] * (1/layers[i].neurons);
        else if (layers[i+1].activation == "Linear")
            *Mb[i]=*Mb[i] * sqrt(2/layers[i].neurons);
        else if (layers[i+1].activation == "satlinear2")
            {*Mb[i]=*Mb[i] * (1/layers[i].neurons);}
        else if (layers[i+1].activation == "satlinear3")
            {*Mb[i]=*Mb[i] * (1/layers[i].neurons);}


        parameters.put(CharGen("b", i + 1), *Mb[i]);
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void NeuralNetwork::train(const Matrix& X, Matrix& Y, float alpha, int numOfEpochs, int minibatchSize, string optimizer1,int Numprint, string ET,float lambda,bool batchNorm)
{
    ErrorType = ET;
    optimizer = optimizer1;
    float COST=1000;
    Matrix CostArr(numOfEpochs, 1);                                 // Array of costs calculated at every epoch
    Matrix Y_hat(Y.Rows(),Y.Columns());                              // Activations of last layer
    Matrix Y_hat_classified(Y.Rows(),Y.Columns());                   // Activations of last layer after classification
    int t = 0;                                                             // Counter used for Adam and LM optimizers
    if(batchNorm)
    {
        Matrix gamma1(numOfLayers-1,1);
        gamma1.Fill(1);
        Matrix gamma2(numOfLayers-1,1);
        parameters.put("gamma1",gamma1);
        parameters.put("gamma2",gamma2);
    }

    /*ADAM INITIALIZATION*/
    Matrix** Msdw = new Matrix*[numOfLayers - 1];
    Matrix** Mvdw = new Matrix*[numOfLayers - 1];
    Matrix** Msdb = new Matrix*[numOfLayers - 1];
    Matrix** Mvdb = new Matrix*[numOfLayers - 1];
    Matrix sdgamma1(numOfLayers-1,1);
    Matrix sdgamma2(numOfLayers-1,1);
    Matrix vdgamma1(numOfLayers-1,1);
    Matrix vdgamma2(numOfLayers-1,1);
    if (optimizer == "Adam")
    {
        for (int i = 0; i < numOfLayers - 1; i++)                                 // L-1 = number of hidden layers + output layer
        {
            Msdw[i]=new Matrix(layers[i+1].neurons, layers[i].neurons);
            grades.put(CharGen("Sdw", i + 1),*Msdw[i]);

            Mvdw[i]=new Matrix(layers[i+1].neurons, layers[i].neurons);
            grades.put(CharGen("Vdw", i + 1),*Mvdw[i]);

            Msdb[i]=new Matrix(layers[i+1].neurons, 1);
            grades.put(CharGen("Sdb", i + 1),*Msdb[i]);

            Mvdb[i]=new Matrix(layers[i+1].neurons, 1);
            grades.put(CharGen("Vdb", i + 1),*Mvdb[i]);
        }
        if(batchNorm)
        {
            grades.put("sdgamma1",sdgamma1);
            grades.put("vdgamma1",vdgamma1);
            grades.put("sdgamma2",sdgamma2);
            grades.put("vdgamma2",vdgamma2);
        }
    }
    /*END OF ADAM INITIALIZATION*/


    /*INITIALIZATION OF PARAMETER CANCELING TECHNIQUE*/
    momentum = true;
    float alpha0 = alpha;
    dictionary prevParameters;
    prevParameters.setName("prevParameters");
    int s=parameters.size();
    for(int i=0; i<numOfLayers-1; i++)
    {
        Matrix W=parameters[CharGen("W", i + 1)];
        prevParameters.put(CharGen("W", i + 1),W);

        Matrix b=parameters[CharGen("b", i + 1)];
        prevParameters.put(CharGen("b", i + 1),b);
    }
    /*END OF INITIALIZATION OF PARAMETER CANCELING TECHNIQUE*/


    /*BEGINNING OF EPOCHS ITERATIONS*/
    for (int i = 0; i<numOfEpochs; i++)
    {
        /*LM Initialization*/
        Matrix* Q;
        Matrix* g;
        if(optimizer == "LM")
        {
            minibatchSize = 1;
            int WSIZE = 0;
            for (int i = 0; i < numOfLayers - 1; i++)                                 // L-1 = number of hidden layers + output layer
                WSIZE += layers[i+1].neurons * layers[i].neurons + layers[i+1].neurons * 1;
            Q = new Matrix(WSIZE,WSIZE);
            g = new Matrix(WSIZE,1);
        }
        /*End OF LM Initialization*/

        /*Iterations on mini batches*/
        int m = X.Columns();
        int numOfMiniBatches = m / minibatchSize;
        for (int j = 0; j<numOfMiniBatches; j++)
        {

            Matrix cur_X = X(0, j*minibatchSize, X.Rows() - 1, ((j + 1)*(minibatchSize)-1));
            Matrix cur_Y = Y(0, j*minibatchSize, Y.Rows() - 1, ((j + 1)*(minibatchSize)-1));
            cache.put(CharGen("A", 0), cur_X);
            Y_hat = feedforward(cur_X, layers, numOfLayers,batchNorm);
            if (optimizer == "LM")
                BackProp(cur_X, cur_Y, Y_hat, alpha0, layers, numOfLayers, j, *Q, *g, m,lambda,batchNorm);
            else
            {
                BackProp(cur_X, cur_Y, Y_hat, alpha0, layers, numOfLayers, t, *Q, *g, m,lambda,batchNorm);
                t++;
            }
            cache.clear();
        }
        /*End of mini batch iterations*/

        /*Calculating cost for 1 epoch*/
        if(i%Numprint==0)
        {
            Y_hat = feedforward(X, layers,numOfLayers,batchNorm);
            COST = numOfErrs(&Y_hat,&Y);
            if(i%10==0) cout<<"Iteration number" <<setw(4)<< i << ":" << "    #numOfErrs="<<setw(4)<< COST <<"    #maxErr="<<setw(9)<<maxErr<<"    #cost="<<setw(12)<<CostFunc(Y,Y_hat,0,numOfLayers)<<endl;

        }
        /*End of calculating cost*/

        /*Parameter canceling technique*/
        if(i == -1)
        {
            float CostSlope = (CostArr.access(i-1, 0) - CostArr.access(i, 0)) / CostArr.access(i-1, 0);

            //In case of LM optimizer
            if(optimizer == "LM")
            {
                if (CostSlope <= 0)
                {
                    if(t <= 5)
                    {
                        t++;
                        alpha0 *= 10;
                        int s=parameters.size();
                        for(int i=0; i<s/2; i++)
                        {
                            parameters.erase(CharGen("W",i+1));
                            Matrix W=prevParameters[CharGen("W",i+1)];
                            parameters.put(CharGen("W",i+1),W);

                            parameters.erase(CharGen("b",i+1));
                            Matrix b=prevParameters[CharGen("b",i+1)];
                            parameters.put(CharGen("b",i+1),b);
                        }
                    }
                }
                else
                    alpha0 /= 10;
            }
            //End of case of LM optimizer

            //In case of Adam optimizer
            if(i == 1000000000000000000)
            {
                momentum = true;
                float CostSlope = (CostArr.access(i-1, 0) - CostArr.access(i, 0)) / CostArr.access(i-1, 0);
                if (CostSlope <= 0)
                {
                    momentum = false;
                    alpha0 = 0.995*alpha0;
                    if(CostSlope<-.01)
                    {
                        int s=parameters.size();
                        for(int i=0; i<s/2; i++)
                        {
                            parameters.erase(CharGen("W",i+1));
                            Matrix W=prevParameters[CharGen("W",i+1)];
                            parameters.put(CharGen("W",i+1),W);

                            parameters.erase(CharGen("b",i+1));
                            Matrix b=prevParameters[CharGen("b",i+1)];
                            parameters.put(CharGen("b",i+1),b);
                        }
                    }

                }
                else
                {
                    momentum = true;
                    alpha0 = (1/0.995)*alpha0;
                    for(int i=0; i<s/2; i++)
                    {
                        prevParameters.erase(CharGen("W",i+1));
                        Matrix W=parameters[CharGen("W",i+1)];
                        prevParameters.put(CharGen("W",i+1),W);
                        prevParameters.erase(CharGen("b",i+1));
                        Matrix b=parameters[CharGen("b",i+1)];
                        prevParameters.put(CharGen("b",i+1),b);
                    }
                }
            }
            //End of case of Adam optimizer
        }
        /*End of Parameter canceling technique*/

        /*Clearing A and z of every layer*/
        cache.clear();
        /*End of clearing A and z*/

        /*Clearing LM parameters Q and g*/
        if(optimizer == "LM")
        {
            delete Q;
            delete g;
        }
        /*End of clearing LM parameters Q and g*/

        if (maxErr == 0)
        {
           cout<<"Iteration number" <<setw(4)<< i << ":" << "    #numOfErrs="<<setw(4)<< COST <<"    #maxErr="<<setw(9)<<maxErr<<"    #cost="<<setw(12)<<CostFunc(Y,Y_hat,0,numOfLayers)<<endl;
           break;
        }
    }
    /*END OF EPOCHS ITERATIONS*/


    /*CLASSIFICATION AND ACCURACY TESTING OF ACTIVATIONS*/
    Y_hat = feedforward(X, layers,numOfLayers,batchNorm);
    Y_hat_classified=classify(Y_hat);
    AccuracyTest(Y, Y_hat_classified, X, layers, numOfLayers);
    /*END OF CLASSIFICATION AND ACCURACY TESTING OF ACTIVATIONS*/


    /*CLEARING sdw, vdw, sdb, vdb*/
    if(optimizer == "Adam")
    {
        for(int i = 0; i < numOfLayers - 1; i++)
        {
            grades.erase(CharGen("Sdw", i + 1));
            delete Msdw[i];

            grades.erase(CharGen("Vdw", i + 1));
            delete Mvdw[i];

            grades.erase(CharGen("Sdb", i + 1));
            delete Msdb[i];

            grades.erase(CharGen("Vdb", i + 1));
            delete Mvdb[i];
        }
        if(batchNorm)
        {
            grades.erase("sdgamma1");
            grades.erase("sdgamma2");
            grades.erase("vdgamma1");
            grades.erase("vdgamma2");
        }
    }

    /*END OF CLEARING sdw, vdw, sdb, vdb*/
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void NeuralNetwork::test(Matrix X_test, Matrix Y, string path,bool batchNorm)
{
    cache.clear();
    Matrix Y_hat =feedforward(X_test,layers, numOfLayers,batchNorm);
    Matrix yhat_classified=classify(Y_hat);
    storedata(X_test,Y,Y_hat,yhat_classified,path);
}
//////////////////////////////////////////////////////////*********************************\/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////|*****PRIVATE MEMBER FUNCTIONS*****|////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////\*********************************//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix NeuralNetwork::feedforward(const Matrix& x, layer* layers,int L,bool batchNorm)
{
    /*Initialization for parameters needed in batch norm*/
    Matrix gamma1(1,1);   //gamma for each layer
    Matrix gamma2(1,1);   //beta for each layer
    Matrix mean(L-1,1);   //mean of z for each layer
    Matrix sd(L-1,1);     //standard deviation of z for each layer
    Matrix z_telda(1,1);  //z-mean of z
    Matrix z_new(1,1);    //z after normalization,scaling and shifting by gamma and beta
    float eps=1e-7;             //to make sure that we don`t divide by zero
    /////////////////////////////////////////
    if(batchNorm)
    {
        gamma1=parameters["gamma1"];
        gamma2=parameters["gamma2"];
    }

    for(int i=0; i<L-1; i++)
    {

        Matrix W=parameters[CharGen("W", i + 1)];
        Matrix b=parameters[CharGen("b", i + 1)];

        Matrix z(1,1);
        Matrix A(1,1);

        if(i==0)
        {
            if(batchNorm)
            {
                z=W.dot(x);
                mean.access(i,0) = z.sumall() / (z.Rows() * z.Columns());
                z_telda = z - mean.access(i,0);
                sd.access(i,0)=(z_telda.square()).sumall() / (z.Rows() * z.Columns());
                z_telda = z_telda / sqrt(sd.access(i,0)+eps);
                z_new=z_telda*gamma1.access(i,0)+gamma2.access(i,0);
                cache.put(CharGen("zt",i+1),z_telda);
                cache.put(CharGen("zn",i+1),z_new);
            }
            else
                z=W.dot(x)+b;

            cache.put(CharGen("z", i + 1),z);
        }
        else
        {
            Matrix Aprev=cache[CharGen("A", i)];
            if(batchNorm)
            {
                z=W.dot(Aprev);
                mean.access(i,0) = z.sumall() / (z.Rows() * z.Columns());
                z_telda = z - mean.access(i,0);
                sd.access(i,0)=(z_telda.square()).sumall() / (z.Rows() * z.Columns());
                z_telda = z_telda / sqrt(sd.access(i,0)+eps);
                z_new=z_telda*gamma1.access(i,0)+gamma2.access(i,0);
                cache.put(CharGen("zt",i+1),z_telda);
                cache.put(CharGen("zn",i+1),z_new);
            }
            else
                z=W.dot(Aprev)+b;
            cache.put(CharGen("z", i + 1),z);

        }

        string activation=layers[i+1].activation;
        if(batchNorm)
        {

            if(activation=="relu")
                A=relu(z_new);
            else if (activation=="tanh")
                A=mytanh(z_new);
            else if(activation=="sigmoid")
                A=sigmoid(z_new);
            else if(activation=="satLinear")
                A=satLinear(z_new);
            else if(activation=="satlinear2")
                A=satLinear2(z_new,maxErr);
            else if(activation=="Linear")
                A=Linear(z_new);
            else if(activation=="satlinear3")
                A=satLinear3(z_new,maxErr);
        }
        else
        {
            if(activation=="relu")
                A=relu(z);
            else if (activation=="tanh")
                A=mytanh(z);
            else if(activation=="sigmoid")
                A=sigmoid(z);
            else if(activation=="satLinear")
                A=satLinear(z);
            else if(activation=="satlinear2")
                A=satLinear2(z,maxErr);
            else if(activation=="Linear")
                A=Linear(z);
            else if(activation=="satlinear3")
                A=satLinear3(z,maxErr);
        }
        cache.put(CharGen("A", i + 1),A);
    }
    if(batchNorm)
    {
        cache.put("mean",mean);
        cache.put("sd",sd);
    }

    Matrix yhat=cache[CharGen("A", L - 1)];
    return yhat;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void NeuralNetwork::calGrads(const Matrix& X, const Matrix& Y, const Matrix& Y_hat, layer* layers, int L,float lambda,bool batchNorm)
{
    float m = X.Columns();
    /*Initialization of parameters for batch Norm*/
    Matrix dgamma1(numOfLayers-1,1);
    Matrix dgamma2(numOfLayers-1,1);
    Matrix gamma1(1,1);
    Matrix gamma2(1,1);
    Matrix mean(1,1);
    Matrix sd(1,1);
    float eps=1e-7;
    Matrix dbLast(1,1);
    Matrix dbl(1,1);
    Matrix db1(1,1);
    Matrix b(1,1);
    if(batchNorm)
    {
        gamma1=parameters["gamma1"];
        gamma2=parameters["gamma2"];
        mean=cache["mean"];
        sd=cache["sd"];
    }
    /*End of initialization for batch norm*/

    /*Output layer*/
    Matrix dzLast(1, 1);
    Matrix zLast = cache[CharGen("z", L - 1)];
    if (ErrorType == "SquareErr")
    {
        Matrix dAprevLast = (Y_hat - Y) / m;
        if (optimizer == "LM")
        {
            grades.put("e", dAprevLast);
            dAprevLast.Fill(1);
        }
        if (layers[L-1].activation == "sigmoid")
            dzLast = dAprevLast * dsigmoid(zLast);
        else if (layers[L-1].activation == "tanh")
            dzLast = dAprevLast * dtanh(zLast);
        else if (layers[L-1].activation == "relu")
            dzLast = dAprevLast * drelu(zLast);
        else if (layers[L-1].activation == "leakyRelu")
            dzLast = dAprevLast * dleakyRelu(zLast);
        else if (layers[L-1].activation == "satLinear")
            dzLast = dAprevLast * dsatLinear(zLast);
        else if (layers[L-1].activation == "Linear")
            dzLast = dAprevLast * dLinear(zLast);
        else if (layers[L-1].activation == "satlinear2")
            dzLast = dAprevLast * dsatLinear2(zLast,maxErr);
        else if (layers[L-1].activation == "satlinear3")
            dzLast = dAprevLast * dsatLinear3(zLast,maxErr);
    }
    else if (ErrorType == "CrossEntropy")
        dzLast = Y_hat - Y;

    if (batchNorm)
    {
        //dzlast means dznew
        Matrix z_telda=cache[CharGen("zt",L-1)];
        dgamma1.access(L-2,0)=(dzLast*z_telda).sumall();
        dgamma2.access(L-2,0)=dzLast.sumall();
        Matrix dz_telda=dzLast*gamma1.access(L-2,0);
        float d=-0.5/pow((sd.access(L-2,0)+eps),1.5);
        Matrix temp1=zLast-mean.access(L-2,0);
        float dsd=((temp1*d)*dz_telda).sumall();
        float ivar=1/sqrt(sd.access(L-2,0)+eps);
        float dmean=(dz_telda*(-1*ivar)).sumall();
        dmean+=(dsd/m)*((temp1*2).sumall());
        dzLast=dz_telda*ivar;
        dzLast=dzLast+temp1*(2*dsd/m);
        dzLast=dzLast+(dmean/m);
    }
    Matrix AprevLast = cache[CharGen("A", L - 2)];
    Matrix dWLast = dzLast.dot(AprevLast.transpose()) / m;
    if(!batchNorm)
        dbLast = dzLast.sum("column") / m;
    Matrix WLast = parameters[CharGen("W", L - 1)];
    Matrix WLast_trans=WLast.transpose();
    Matrix dAprevLast = WLast_trans.dot(dzLast);
    if(lambda !=0)
    {
        dWLast=dWLast+WLast*(lambda/m);
    }

    grades.put(CharGen("dW", L - 1), dWLast);

    if(!batchNorm)
        grades.put(CharGen("db", L - 1), dbLast);

    grades.put(CharGen("dA", L - 2), dAprevLast);
    /*End of output layer*/

    /*Hidden layers*/
    for(int i = L-2; i > 0; i--)
    {

        Matrix dAl = grades[CharGen("dA", i)];
        grades.erase(CharGen("dA", i));

        Matrix zl = cache[CharGen("z", i)];
        Matrix dzl(1, 1);
        if (layers[i].activation == "sigmoid")
            dzl = dAl * dsigmoid(zl);
        else if (layers[i].activation == "tanh")
            dzl = dAl * dtanh(zl);
        else if (layers[i].activation == "relu")
            dzl = dAl * drelu(zl);
        else if (layers[i].activation == "leakyRelu")
            dzl = dAl * dleakyRelu(zl);
        else if (layers[i].activation == "satLinear")
            dzl = dAl * dsatLinear(zl);
        else if (layers[i].activation == "Linear")
            dzl = dAl * dLinear(zl);
        else if (layers[i].activation == "satlinear2")
            dzl = dAl * dsatLinear2(zl,maxErr);
        else if (layers[i].activation == "satlinear3")
            dzl = dAl * dsatLinear3(zl,maxErr);

        if (batchNorm)
        {
            //dzl means dznew
            Matrix z_telda=cache[CharGen("zt",i)];
            dgamma1.access(i-1,0)=(dzl*z_telda).sumall();
            dgamma2.access(i-1,0)=dzl.sumall();
            Matrix dz_telda=dzl*gamma1.access(i-1,0);
            float d=-0.5/pow((sd.access(i-1,0)+eps),1.5);
            Matrix temp1=zl-mean.access(i-1,0);
            float dsd=((temp1*d)*dz_telda).sumall();
            float ivar=1/sqrt(sd.access(i-1,0)+eps);
            float dmean=(dz_telda*(-1*ivar)).sumall();
            dmean+=(dsd/m)*((temp1*2).sumall());
            dzl=dz_telda*ivar;
            dzl=dzl+temp1*(2*dsd/m);
            dzl=dzl+(dmean/m);
        }
        Matrix Al_1 = cache[CharGen("A", i - 1)];
        Matrix dWl = dzl.dot(Al_1.transpose()) / m;
        if(!batchNorm)
            dbl = dzl.sum("column") / m;

        Matrix Wl = parameters[CharGen("W", i)];
        Matrix Wl_trans=Wl.transpose();
        Matrix dAl_1 = Wl_trans.dot(dzl);

        if(lambda!=0)
        {
            dWl=dWl+Wl*(lambda/m);
        }

        grades.put(CharGen("dW", i), dWl);

        if(!batchNorm)
            grades.put(CharGen("db", i), dbl);

        if(i != 1)
            grades.put(CharGen("dA", i - 1), dAl_1);
    }
    /*End of hidden layers*/
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void NeuralNetwork::updateParameters(float& alpha, layer* layers, int L, int iteration, Matrix& Q, Matrix& g, int m,bool batchNorm)
{
    /*START OF GRADIENT DESCENT OPTIMIZER*/
    if (optimizer == "GradientDescent")
    {
        if(batchNorm)
        {
            Matrix gamma1=parameters["gamma1"];
            parameters.erase("gamma1");
            Matrix dgamma1=grades["dgamma1"];
            gamma1=gamma1-dgamma1*alpha;
            parameters.put("gamma1",gamma1);

            Matrix gamma2=parameters["gamma2"];
            parameters.erase("gamma2");
            Matrix dgamma2=grades["dgamma2"];
            gamma2=gamma2-dgamma2*alpha;
            parameters.put("gamma2",gamma2);

            grades.erase("dgamma1");
            grades.erase("dgamma2");
        }

        for(int i = 0; i < L-1; i++)
        {
            Matrix Wul = parameters[CharGen("W", i + 1)] - grades[CharGen("dW", i + 1)] * alpha;
            parameters.erase(CharGen("W", i + 1));
            parameters.put(CharGen("W", i + 1),Wul);
            grades.erase(CharGen("dW", i + 1));
            if(!batchNorm)
            {
                Matrix bul = parameters[CharGen("b", i + 1)] - grades[CharGen("db", i + 1)] * alpha;
                parameters.erase(CharGen("b", i + 1));
                parameters.put(CharGen("b", i + 1),bul);
                grades.erase(CharGen("db", i + 1));
            }
        }
    }

    /*END OF GRADIENT DESCENT OPTIMIZER*/

    /*START OF ADAM OPTIMIZER*/
    else if(optimizer == "Adam")
    {
        float beta1 = 0.9;
        float beta2 = 0.999;
        float epsilon = 1e-8;
        Matrix Vdb(1,1);
        Matrix Sdb(1,1);
        Matrix db(1,1);
        Matrix b(1,1);
        Matrix Sdb_corr(1,1);
        Matrix Vdb_corr(1,1);
        for(int i = 0; i < L-1; i++)
        {
            /*Getting variables from dictionaries*/
            Matrix Vdw = grades[CharGen("Vdw", i + 1)];
            Matrix Sdw = grades[CharGen("Sdw", i + 1)];
            if(!batchNorm)
            {
                Vdb = grades[CharGen("Vdb", i + 1)];
                Sdb = grades[CharGen("Sdb", i + 1)];
            }

            Matrix dW = grades[CharGen("dW", i + 1)];
            Matrix W = parameters[CharGen("W", i + 1)];
            if(!batchNorm)
            {
                db = grades[CharGen("db", i + 1)];
                b = parameters[CharGen("b", i + 1)];
            }

            /*Updating Vdw, Vdb, Sdw, Sdb*/
            Vdw = (Vdw * (beta1 * momentum)) + (dW * (1-beta1 * momentum));
            grades.erase(CharGen("Vdw", i + 1));
            grades.put(CharGen("Vdw", i + 1), Vdw);

            if(!batchNorm)
            {
                Vdb = (Vdb * (beta1 * momentum)) + (db * (1-beta1 * momentum));
                grades.erase(CharGen("Vdb", i + 1));
                grades.put(CharGen("Vdb", i + 1), Vdb);
            }

            Sdw = (Sdw * beta2) + (dW.square() * (1-beta2));
            grades.erase(CharGen("Sdw", i + 1));
            grades.put(CharGen("Sdw", i + 1), Sdw);

            if(!batchNorm)
            {
                Sdb = (Sdb * beta2) + (db.square() * (1-beta2));
                grades.erase(CharGen("Sdb", i + 1));
                grades.put(CharGen("Sdb", i + 1), Sdb);
            }

            /*Correcting first iterations*/
            Matrix Vdw_corr = Vdw / (1 - pow(beta1, iteration+1));
            Matrix Sdw_corr = Sdw / (1 - pow(beta2, iteration+1));
            if(!batchNorm)
            {
                Sdb_corr = Sdb / (1 - pow(beta2, iteration+1));
                Vdb_corr = Vdb / (1 - pow(beta1, iteration+1));
            }
            /*Updating parameters*/
            Matrix temp1 = Vdw_corr / (Sdw_corr.Sqrt() + epsilon);
            Matrix Wu = W - temp1 * alpha;
            parameters.erase(CharGen("W", i + 1));
            parameters.put(CharGen("W", i + 1), Wu);

            if(!batchNorm)
            {
                Matrix temp2 = Vdb_corr / (Sdb_corr.Sqrt() + epsilon);
                Matrix bu = b - temp2 * alpha;
                parameters.erase(CharGen("b", i + 1));
                parameters.put(CharGen("b", i + 1), bu);
            }

            /*Erasing dW, db*/
            grades.erase(CharGen("dW", i + 1));
            if(!batchNorm)
                grades.erase(CharGen("db", i + 1));
        }
        if(batchNorm)
        {
            /*Getting variables from dictionaries*/
            Matrix vdgamma1 = grades["vdgamma1"];
            Matrix sdgamma1 = grades["sdgamma1"];
            Matrix vdgamma2 = grades["vdgamma2"];
            Matrix sdgamma2 = grades["sdgamma2"];
            Matrix dgamma1  = grades["dgamma1"];
            Matrix dgamma2  = grades["dgamma2"];
            Matrix gamma1   = parameters["gamma1"];
            Matrix gamma2   = parameters["gamma2"];
            /*Updating vdgamma1, vdgamma2, sdgamma1, sdgamma2*/
            vdgamma1 = (vdgamma1 * (beta1 * momentum)) + (dgamma1 * (1-beta1 * momentum));
            grades.erase("vdgamma1");
            grades.put("vdgamma1", vdgamma1);

            vdgamma2 = (vdgamma2 * (beta1 * momentum)) + (dgamma2 * (1-beta1 * momentum));
            grades.erase("vdgamma2");
            grades.put("vdgamma2", vdgamma2);

            sdgamma1 = (sdgamma1 * beta2) + (dgamma1.square() * (1-beta2));
            grades.erase("sdgamma1");
            grades.put("sdgamma1", sdgamma1);

            sdgamma2 = (sdgamma2 * beta2) + (dgamma2.square() * (1-beta2));
            grades.erase("sdgamma2");
            grades.put("sdgamma2",sdgamma2);
            /*Correcting first iterations*/
            Matrix vdgamma1_corr = vdgamma1 / (1 - pow(beta1, iteration+1));
            Matrix vdgamma2_corr = vdgamma2 / (1 - pow(beta1, iteration+1));
            Matrix sdgamma1_corr = sdgamma1 / (1 - pow(beta2, iteration+1));
            Matrix sdgamma2_corr = sdgamma2 / (1 - pow(beta2, iteration+1));
            /*Updating parameters*/
            Matrix temp1 = vdgamma1_corr / (sdgamma1_corr.Sqrt() + epsilon);
            Matrix gamma1u = gamma1 - temp1 * alpha;
            parameters.erase("gamma1");
            parameters.put("gamma1", gamma1u);

            Matrix temp2 = vdgamma2_corr / (sdgamma2_corr.Sqrt() + epsilon);
            Matrix gamma2u = gamma2 - temp2 * alpha;
            parameters.erase("gamma2");
            parameters.put("gamma2", gamma2u);

            /*Erasing dgamma1, dgamma2*/
            grades.erase("dgamma1");
            grades.erase("dgamma2");
        }
    }
    /*END OF ADAM OPTIMIZER*/

    /*START OF LM OPTIMIZER*/
    else if(optimizer == "LM")
    {
        int WSIZE = Q.Rows();
        /*Initializing j (the differentiation of error w.r.t all parameters)*/
        Matrix j(1, WSIZE);
        int counter = 0;
        for(int i = 0; i < L - 1; i++)
        {
            Matrix dW = grades[CharGen("dW", i+1)];
            for(int p=0; p<dW.Rows(); p++)
                for(int k=0; k<dW.Columns(); k++)
                {
                    j.access(0,counter) = dW.access(p,k);
                    counter++;
                }

            Matrix db = grades[CharGen("db", i+1)];
            for(int p=0; p<db.Rows(); p++)
                for(int k=0; k<db.Columns(); k++)
                {
                    j.access(0,counter) = db.access(p,k);
                    counter++;
                }
        }
        /*End of initializing j*/

        /*Updating Q and g*/
        Q = Q + j.transpose().dot(j);
        g = g + j.transpose().dot(grades["e"]);
        grades.clear();
        /*End of updating Q and g*/

        /*Update W's and b's after all patterns*/
        if(iteration == m-1)
        {
            Matrix W(WSIZE, 1);
            counter = 0;
            for(int i = 0; i < L - 1; i++)
            {
                Matrix Wi = parameters[CharGen("W", i+1)];
                for(int p=0; p<Wi.Rows(); p++)
                    for(int k=0; k<Wi.Columns(); k++)
                    {
                        W.access(counter,0) = Wi.access(p,k);
                        counter++;
                    }

                Matrix bi = parameters[CharGen("b", i+1)];
                for(int p=0; p<bi.Rows(); p++)
                    for(int k=0; k<bi.Columns(); k++)
                    {
                        W.access(counter,0) = bi.access(p,k);
                        counter++;
                    }
            }

            Matrix I(WSIZE, WSIZE, Identity);
            Matrix dW = I * alpha;
            dW = Q + dW;
            dW = dW.Inverse();
            dW = dW.dot(g);
            W = W + dW;

            counter = 0;
            for(int i = 0; i < L - 1; i++)
            {
                Matrix Wi = parameters[CharGen("W", i+1)];
                for(int p=0; p<Wi.Rows(); p++)
                    for(int k=0; k<Wi.Columns(); k++)
                    {
                        Wi.access(p,k) = W.access(counter,0);
                        counter++;
                    }
                parameters.erase(CharGen("W", i+1));
                parameters.put(CharGen("W", i+1), Wi);

                Matrix bi = parameters[CharGen("b", i+1)];
                for(int p=0; p<bi.Rows(); p++)
                    for(int k=0; k<bi.Columns(); k++)
                    {
                        bi.access(p,k) = W.access(counter,0);
                        counter++;
                    }
                parameters.erase(CharGen("b", i+1));
                parameters.put(CharGen("b", i+1), bi);
            }
        }
        /*End of updating W's and b's*/
    }
    /*END OF LM OPTIMIZER*/


}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void NeuralNetwork::BackProp(const Matrix& X, const Matrix& Y, const Matrix& Y_hat, float& alpha, layer* layers, int L, int iteration, Matrix& Q, Matrix& g, int m,float lambda,bool batchNorm)
{
    calGrads(X, Y, Y_hat, layers, L,lambda,batchNorm);
    updateParameters(alpha, layers, L, iteration, Q, g, m,batchNorm);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix NeuralNetwork::costMul(Matrix Y, Matrix Y_hat)
{
    Matrix result(Y.Rows(), Y.Columns());
    for (int i = 0; i < Y.Rows(); i++)
        for (int j = 0; j < Y.Columns(); j++)
        {
            float epsilon = 1e-7;
            if (Y.access(i, j) == 0)
                result.access(i, j) = log(1 + epsilon - Y_hat.access(i, j));
            else
                result.access(i, j) = log(Y_hat.access(i, j) + epsilon);
        }
    return result;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
float NeuralNetwork::CostFunc (const Matrix& y,Matrix& yhat,float lambda,int L)
{
    float cost;
    float m=y.Columns();
    if(ErrorType=="CrossEntropy")
    {
        Matrix result = costMul(y,yhat);
        cost=(-1/m)*(result.sumall());
    }

    else
    {
        Matrix result=yhat-y;
        result = result.square();
        cost=(1/(2*m)*(result.sumall()));
    }
    if(lambda!=0)
    {
        float regulization_cost=0;
        for(int i=0; i<L-1; i++)
        {
            Matrix W=parameters[CharGen("W",i+1)];
            W=W.square();
            regulization_cost+=W.sumall();
        }
        regulization_cost=regulization_cost*(lambda/(2*m));
        cost+=regulization_cost;
    }
    return cost;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix NeuralNetwork::classify(Matrix Y_hat)
{
    Matrix yhat_classified(Y_hat.Rows(),Y_hat.Columns());

    if (ErrorType == "SquareErr")
    {
        for(int i=0; i<Y_hat.Columns(); i++)                  //making the predictions either 1 or 0
        {
            if(Y_hat.access(0,i)<=-.9)
                yhat_classified.access(0,i)=-1;
            else if(Y_hat.access(0,i)>=.9)
                yhat_classified.access(0,i)=1;
            else
                yhat_classified.access(0,i)=-10;
        }
    }
    else if (ErrorType == "CrossEntropy")
    {
        for(int i=0; i<Y_hat.Columns(); i++)                  //making the predictions either 1 or 0
        {
            if(Y_hat.access(0,i)<=.2)
                yhat_classified.access(0,i)=0;
            else if(Y_hat.access(0,i)>=.8)
                yhat_classified.access(0,i)=1;
            else
                yhat_classified.access(0,i)=-10;
        }
    }

    return yhat_classified;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void NeuralNetwork::AccuracyTest(const Matrix& Y,Matrix& Y_hat_classified, const Matrix& X, layer*layers, int L)
{
    Matrix diff = Y - Y_hat_classified;                     //the ones in diff matrix represent the false predictions
    float errSum = 0;
    float uncertain = 0;

    if (ErrorType == "CrossEntropy")
    {
        for(int i=0; i<diff.Columns(); i++)
        {
            if(diff.access(0,i)<0)
                diff.access(0,i)=diff.access(0,i)*-1;

            if(diff.access(0,i)!=10 && diff.access(0,i)!=11)
                errSum=errSum+diff.access(0,i);
            else
                uncertain++;
        }
    }
    else if (ErrorType == "SquareErr")
    {
        for(int i=0; i<diff.Columns(); i++)
        {
            if(diff.access(0,i)<0)
                diff.access(0,i)=diff.access(0,i)*-1;

            if(diff.access(0,i)!=11 && diff.access(0,i)!=9)
                errSum=errSum+diff.access(0,i);
            else
                uncertain++;
        }
        errSum = errSum / 2;
    }

    cout <<endl<< "The maximum Err = "<<maxErr<<endl;
    cout <<endl<< "The number of false predictions = "<<errSum<<endl;
    cout <<endl<< "The number of uncertain predictions = "<<uncertain<<endl;
    float Accur=1-((errSum+uncertain)/diff.Columns());
    cout<<endl<<"Accuracy = "<<Accur*100<<" %"<<endl<<endl;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void NeuralNetwork::storedata(Matrix X, Matrix Y, Matrix Yhat, Matrix yhat_classified, string path)
{
    ofstream outfile(path.data());
    for(int i=0; i<X.Columns(); i++)
    {
        outfile<<i<<") ";
        for(int j=0; j<X.Rows(); j++)
        {
            float a=X.access(j,i);
            outfile<<a<<" ";

        }
        float b=Y.access(0,i);
        float c=Yhat.access(0,i);
        float d=yhat_classified.access(0,i);
        outfile<<"/ "<<b<<"/ "<<c<<"/ "<<d<<endl<<endl;
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void NeuralNetwork::print()
{
    parameters.print();
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*Matrix NeuralNetwork::Classify_Train(Matrix Y_hat)
{
    Matrix yhat_classified(Y_hat.Rows(),Y_hat.Columns());

    if (ErrorType == "SquareErr")
    {
        for(int i=0; i<Y_hat.Columns(); i++)                  //making the predictions either 1 or 0
        {
            if(Y_hat.access(0,i)<=-.9)
                yhat_classified.access(0,i)=-1;
            else if(Y_hat.access(0,i)>=.9)
                yhat_classified.access(0,i)=1;
        }
    }
    else if (ErrorType == "CrossEntropy")
    {
        for(int i=0; i<Y_hat.Columns(); i++)                  //making the predictions either 1 or 0
        {
            if(Y_hat.access(0,i)<=.2)
                yhat_classified.access(0,i)=0;
            else if(Y_hat.access(0,i)>=.8)
                yhat_classified.access(0,i)=1;
        }
    }

    return yhat_classified;
}*/
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
float NeuralNetwork::AbsErr(Matrix* Y_hat, Matrix* Y)
{
    float err=0;
    float e=0;
    for(int i=0; i<Y_hat->Rows(); i++)
        for(int j=0; j<Y_hat->Columns(); j++)
        {
            e = Y_hat->access(i,j)-Y->access(i,j);
            if(e < 0)
                err=err-e;
            else
                err=err+e;
        }

    return err;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
float NeuralNetwork::numOfErrs(Matrix* Y_hat, Matrix* Y)
{
    maxErr=0;
    float e=0;
    float errs=0;
    for(int i=0; i<Y_hat->Rows(); i++)
        for(int j=0; j<Y_hat->Columns(); j++)
        {
            e = Y_hat->access(i,j) - Y->access(i,j);

            if(e<0)
                e=e * -1;

            if(e!=0)
                errs++;

            if(e>maxErr)
                maxErr=e;
        }
    return errs;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
