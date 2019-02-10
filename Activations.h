#pragma once

#include "Matrix.h"

#ifndef ACTIVATIONS_H_INCLUDED
#define ACTIVATIONS_H_INCLUDED
float ActivPar = 0.8;
typedef matrix<float> Matrix;
////////////////////////////////////////////
Matrix sigmoid(Matrix& z)
{
    int r=z.Rows();
    int c=z.Columns();
    Matrix A(r,c);
    for(int i=0;i<r;i++)
    {
        for(int j=0;j<c;j++)
        {
            A.access(i,j) = 1/(1+exp(-1*z.access(i,j)));
        }
    }
    return A;
}
///////////////////////////////////////////
Matrix mytanh(Matrix& z)
{
    int r=z.Rows();
    int c=z.Columns();
    Matrix A(r,c);
    for(int i=0;i<r;i++)
    {
        for(int j=0;j<c;j++)
        {
            A.access(i,j)=tanh(z.access(i,j));
        }
    }
    return A;
}
////////////////////////////////////////
Matrix relu(Matrix& z)
{
    int r=z.Rows();
    int c=z.Columns();
    matrix <float>A(r,c);
    for(int i=0;i<r;i++)
    {
        for(int j=0;j<c;j++)
        {
            if(z.access(i,j)>=0)
            A.access(i,j)=z.access(i,j);
            else
            A.access(i,j)=0;
        }
    }
    return A;
}
//////////////////////////////////////////
Matrix leakyRelu(Matrix& z)
{
    int r=z.Rows();
    int c=z.Columns();
    matrix <float>A(r,c);
    for(int i=0;i<r;i++)
    {
        for(int j=0;j<c;j++)
        {
            if(z.access(i,j)>=0)
            A.access(i,j)=z.access(i,j);
            else
            A.access(i,j)=0.01*z.access(i,j);
        }
    }
    return A;
}
//////////////////////////////////////////
Matrix Linear(Matrix& z)
{
    return z;
}
//////////////////////////////////////////
Matrix satLinear(Matrix& z)
{
    int r=z.Rows();
    int c=z.Columns();
    Matrix A(r,c);
    for(int i=0; i<r; i++)
    {
        for(int j=0; j<c; j++)
        {
            if(z.access(i,j)<=-1)
                 A.access(i,j)=-1;
            else if(z.access(i,j)>-1 && z.access(i,j)<1)
                 A.access(i,j)=1*z.access(i,j);
            else
                 A.access(i,j)=1;
        }
    }
    return A;
}
//////////////////////////////////////////
Matrix satLinear2(Matrix& z, float maxErr)
{
    float slope;

    if(maxErr>=ActivPar)
        slope=1;
    else
        slope=2;

    int r=z.Rows();
    int c=z.Columns();
    Matrix A(r,c);

    for(int i=0; i<r; i++)
    {
        for(int j=0; j<c; j++)
        {
            if(z.access(i,j)<=-1/slope)
                 A.access(i,j)=-1;
            else if(z.access(i,j)>-1/slope && z.access(i,j)<1/slope)
                 A.access(i,j)=slope*z.access(i,j);
            else
                 A.access(i,j)=1;
        }
    }

    return A;
}
//////////////////////////////////////////
Matrix satLinear3(Matrix& z,float maxErr)
{
    float slope;

    if(maxErr>=ActivPar)
        slope=1;
    else
        slope=2;

    int r=z.Rows();
    int c=z.Columns();
    Matrix A(r,c);

    for(int i=0; i<r; i++)
    {
        for(int j=0; j<c; j++)
        {
            if(z.access(i,j)<=-1/slope)
                 A.access(i,j)=-1;
            else if(z.access(i,j)>-1/slope && z.access(i,j)<1/slope)
                 A.access(i,j)=slope*z.access(i,j);
            else
                 A.access(i,j)=1;
        }
    }

    return A;
}
//////////////////////////////////////////
Matrix dsigmoid(Matrix& z)
{
    Matrix a=sigmoid(z);
    Matrix da=a*((a*-1)+1);                        //da=a(1-a)
    return da;
}
/////////////////////////////////////////
Matrix dtanh(Matrix& z)
{
    Matrix a=mytanh(z);
    Matrix da=((a*a)*-1) +1;                       //da=1-a^2
    return da;
}
/////////////////////////////////////////
Matrix drelu(Matrix& z)
{
    int r=z.Rows();
    int c=z.Columns();
    Matrix da(r,c);
    for(int i=0;i<r;i++)
    {
        for (int j=0;j<c;j++)
        {
            if(z.access(i,j)<0)
            da.access(i,j)=0;
            else
            da.access(i,j)=1;
        }
    }
    return da;
}
//////////////////////////////////////////
Matrix dleakyRelu(Matrix& z)
{
    int r=z.Rows();
    int c=z.Columns();
    Matrix da(r,c);
    for(int i=0;i<r;i++)
    {
        for (int j=0;j<c;j++)
        {
            if(z.access(i,j)<0)
            da.access(i,j)=0.01;
            else
            da.access(i,j)=1;
        }
    }
    return da;
}

////////////////////////////////////////
Matrix dLinear(Matrix& z)
{
    int r=z.Rows();
    int c=z.Columns();
    Matrix da(r,c);
    for(int i=0;i<r;i++)
    {
        for (int j=0;j<c;j++)
        {
            da.access(i,j)=1;
        }
    }
    return da;
}
/////////////////////////////////////////
Matrix dsatLinear(Matrix& z)
{
    int r=z.Rows();
    int c=z.Columns();
    Matrix da(r,c);
    for(int i=0;i<r;i++)
    {
        for (int j=0;j<c;j++)
        {
            if(z.access(i,j)<=-1)
                 da.access(i,j)=0;
            else if(z.access(i,j)>-1 && z.access(i,j)<1)
                 da.access(i,j)=1;
            else
                 da.access(i,j)=0;
        }
    }
    return da;
}
//////////////////////////////////////////////////
Matrix dsatLinear2(Matrix& z,float maxErr)
{
    float slope;

    if(maxErr>=ActivPar)
        slope=1;
    else
        slope=2;

    int r=z.Rows();
    int c=z.Columns();
    Matrix da(r,c);
    for(int i=0;i<r;i++)
    {
        for (int j=0;j<c;j++)
        {
            if(z.access(i,j)<=-1/slope)
                 da.access(i,j)=0;

            else if(z.access(i,j)>-1/slope && z.access(i,j)<1/slope)
                 da.access(i,j)=slope;

            else
                 da.access(i,j)=0;
        }
    }
    return da;
}
//////////////////////////////////////////////////
Matrix dsatLinear3(Matrix& z, float maxErr)
{
    float slope;

    if(maxErr>=ActivPar)
        slope=1;
    else
        slope=2;

    int r=z.Rows();
    int c=z.Columns();
    Matrix da(r,c);
    for(int i=0;i<r;i++)
    {
        for (int j=0;j<c;j++)
        {
            if(z.access(i,j)<=-1/slope)
                 da.access(i,j)=0;

            else if(z.access(i,j)>-1/slope && z.access(i,j)<1/slope)
                 da.access(i,j)=slope;

            else
                 da.access(i,j)=0;
        }
    }
    return da;
}
//////////////////////////////////////////////////
#endif // ACTIVATIONS_H_INCLUDED
