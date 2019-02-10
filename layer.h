#ifndef LAYER_H_INCLUDED
#define LAYER_H_INCLUDED
#include <string>

struct layer
{
    float neurons;
    string activation;

    void put(float n, string activ)
    {
        neurons=n;
        activation=activ;
    }
};

#endif // LAYER_H_INCLUDED
