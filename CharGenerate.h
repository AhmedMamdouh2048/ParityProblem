#pragma once

#include <string>

#ifndef CHARGENERATE_H_INCLUDED
#define CHARGENERATE_H_INCLUDED
string CharGen(string name, int i)
{
    int temp = i;
    int counter1;   //number of decimal digits in i

    for(counter1 = 0; temp != 0; counter1++)
        temp = temp / 10;

    int counter2=name.size();   //number of chars in name

    string result;
    if(counter2==1){result="W0";}
    if(counter2==2){result="dW0";}
    if(counter2==3){result="Sdw0";}


    for (unsigned int j = 0; j<name.size(); j++) //copy the name into result
        result[j] = name[j];

    int j = counter1 + counter2 - 1;      //copy the number into result
    temp = i;
    do
    {
        result[j] = '0' + (temp % 10);
        temp = temp / 10;
        j--;
    }while (temp != 0);

    return result;
}


#endif // CHARGENERATE_H_INCLUDED
