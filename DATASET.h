#pragma once

#include "Matrix.h"

#ifndef DATASET_H_INCLUDED
#define DATASET_H_INCLUDED
void DATASET(matrix<float>& X, matrix<float>& Y, int n, matrix<float>& X_DATA, matrix<float>& Y_DATA)
{

	void SWAP(matrix<float>& MAT, int i, int k);      	//swap two columns i,k in MAT

	for (int r = 0; r<X.Rows(); r++)                    //generating the large matrix X_DATA
		for (int c = 0; c<X.Columns(); c++)
			for (int nn = 0; nn<n; nn++)
				X_DATA.access(r, c + nn * X.Columns()) = X.access(r, c);

	for (int r = 0; r<Y.Rows(); r++)                    //generating the large matrix Y_DATA
		for (int c = 0; c<Y.Columns(); c++)
			for (int nn = 0; nn<n; nn++)
				Y_DATA.access(r, c + nn * Y.Columns()) = Y.access(r, c);

	for (int i = 0; i<X_DATA.Columns(); i++)            //shuffling X_DATA and Y_DATA
	{
		srand(time(NULL));
		int s = rand() % X_DATA.Columns();
		SWAP(X_DATA, i, s);
		SWAP(Y_DATA, i, s);
	}
}
/////////////////////////////////////////
void SWAP(matrix<float>& MAT, int i, int k)
{
	matrix<float> temp(MAT.Rows(), 1);
	for (int j = 0; j<MAT.Rows(); j++)
	{
		temp.access(j, 0) = MAT.access(j, i);
		MAT.access(j, i) = MAT.access(j, k);
		MAT.access(j, k) = temp.access(j, 0);
	}
}

#endif // DATASET_H_INCLUDED

