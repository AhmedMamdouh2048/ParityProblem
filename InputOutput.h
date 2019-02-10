#include "Matrix.h"

#ifndef INPUTOUTPUT_H_INCLUDED
#define INPUTOUTPUT_H_INCLUDED

void InputOutput(matrix<float>& x,matrix<float>& y, string ET)
{
    /*CALCULATING X*/
    int xrow=x.Rows();
    int xcol=x.Columns();
    if (ET == "CrossEntropy")
        for(int i=0;i<xrow;i++)
            for(int j=0;j<xcol;j++)
            {
                int a=j/pow(2,i);
                if(a%2==0)
                x.access(i,j)=0;
                else
                x.access(i,j)=1;
            }
    else if (ET == "SquareErr")
        for(int i=0;i<xrow;i++)
            for(int j=0;j<xcol;j++)
            {
                int a=j/pow(2,i);
                if(a%2==0)
                x.access(i,j)=-1;
                else
                x.access(i,j)=1;
            }


    /*CALCULATING Y*/
    int ycol=y.Columns();
    if (ET == "CrossEntropy")
        for(int jj=0;jj<ycol;jj++)
        {
            for(int ii=0;ii<xrow;ii++)
            {
                y.access(0,jj)=y.access(0,jj)+x.access(ii,jj);
            }
            int b=y.access(0,jj);
            if(b%2==0)
                y.access(0,jj)=1;
            else
                y.access(0,jj)=0;
        }
    else if (ET == "SquareErr")
        for(int jj=0;jj<ycol;jj++)
        {
            for(int ii=0;ii<xrow;ii++)
            {
                float temp = x.access(ii, jj);
                if (temp == -1)
                    temp = 0;
                 y.access(0,jj)=y.access(0,jj)+temp;
            }
            int b=y.access(0,jj);
            if(b%2==0)
                y.access(0,jj)=1;
            else
                y.access(0,jj)=-1;
        }
}

#endif // INPUTOUTPUT_H_INCLUDED
