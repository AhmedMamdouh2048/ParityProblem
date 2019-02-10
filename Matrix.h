#include <iostream>
#include <ctime>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <math.h>

using namespace std;

#ifndef matrix_head
#define matrix_head

enum MatrixType {Identity, Random};

template<typename E>
class matrix
{
private:
	int row;
	int column;
	E** data;

public:
    //Constructors and destructor
    matrix();                               //Just for experiment
	matrix(int r, int c);                   //Constructs r x c matrix with nothing in it
	matrix(int r, int c, E e);              //Constructs r x c matrix filled with e
	matrix(int r, int c, MatrixType t);     //Constructs r x c matrix which is either Identity or Random (elements type: integer)
	matrix(const matrix& m);                //Copy constructor
	~matrix();                              //Destructor

	//Visualize data
	void print() const;                                 //Prints all elements of matrix
	char* ToString() const;                             //Converts the matrix into a printable string

	//Access functions
	int Rows() const;                                   //Returns the number of rows in the matrix
	int Columns() const;                                //Returns the number of columns in the matrix
	E& access(int r, int c);                            //Referencing the element [r][c]
	matrix operator() (int r1, int c1, int r2 = -1, int c2 = -1) const;
                                                        //Getting a sub_matrix = BIG_MATRIX(beginning of rows, beginning of columns, end of rows, end of columns);
                                                        //Default values of r2 and c2 are the end of rows and columns

	//Arithmetic operations
	matrix operator+ (const matrix m) const;            //Summing 2 matrices
	matrix operator- (const matrix m) const;            //Subtracting 2 matrices
	matrix operator* (const matrix m) const;            //Element-wise product
	matrix operator/ (const matrix m) const;            //Element-wise division
	matrix operator+ (E n) const;
	matrix operator- (E n) const;
	matrix operator/ (E n) const;
	matrix operator* (E n) const;
	void operator= (const matrix& m);                   //Assignment operator
	matrix dot (const matrix m) const;                  //Dot product
	matrix divide (const matrix divisor) const;         //Matrix division

	//Logic operators
	bool operator== (const matrix m) const;
	bool operator!= (const matrix m) const;
	bool IsIdentity() const;
	bool IsIdempotent() const;                          //Checks if the dot product between the matrix and itself is the same matrix
	bool IsSquare() const;
	bool IsSymmetric() const;
	bool IsUpperTriangle() const;
	bool IsLowerTriangle() const;

	//Matrix operations
	matrix Inverse() const;
	matrix CholeskyInverse() const;
	matrix SlowInverse() const;
	matrix transpose() const;
	matrix sum(string choice) const;               //If choice == "row" , result is 1 x column. If choice == "column", result is row x 1.
	E sumall() const;                                   //Sums all elements in matrix
	E MaxElement() const;
	E MinElement() const;
	E determinant() const;
	void Fill(E e);                                     //Fills all elements with e
	matrix LowerTri() const;
	matrix LTinverse() const;

	//Special operations
	matrix getlog() const;                              //log(matrix)
	matrix square() const;                              //Element-wise square
	matrix Sqrt() const;                                //Element-wise square root
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template<typename E>
matrix<E>::matrix()
{
	char ch;
	cout << "Enter the size of the matrix in the format (mxn): ";
	cin >> row >> ch >> column;
	data = new E*[row];
	for (int i = 0; i < row; i++)
		data[i] = new E[column];

	for (int i = 0; i < row; i++)
	{
		cout << "Enter row No." << i << ": " << endl;
		for (int j = 0; j < column; j++)
			cin >> data[i][j];
	}
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename E>
matrix<E>::matrix(int r, int c)
{
	row = r;
	column = c;
	data = new E*[r];
	for (int i = 0; i < r; i++)
		data[i] = new E[c];

    for (int i = 0; i < r; i++)
		for (int j = 0; j < c; j++)
			data[i][j] = 0;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename E>
matrix<E>::matrix(int r, int c, E e)
{
	row = r;
	column = c;
	data = new E*[r];
	for (int i = 0; i < r; i++)
		data[i] = new E[c];

	for (int i = 0; i < r; i++)
		for (int j = 0; j < c; j++)
			data[i][j] = e;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename E>
matrix<E>::matrix(int r, int c, MatrixType t)
{
	row = r;
	column = c;
	data = new E*[r];
	for (int i = 0; i < r; i++)
		data[i] = new E[c];

	switch (t)
	{
	case Identity:
		for (int i = 0; i < r; i++)
			for (int j = 0; j < c; j++)
			{
				if (i == j)
					data[i][j] = 1;
				else
					data[i][j] = 0;
			}
		break;

	case Random:
		for (int i = 0; i < r; i++)
			for (int j = 0; j < c; j++)
				data[i][j] = rand();
		break;
	}
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename E>
matrix<E>::matrix(const matrix<E>& m)
{
    row = m.row;
    column = m.column;
    data = new E*[row];
    for (int i = 0; i < row; i++)
        data[i] = new E[column];
    for (int i = 0; i < row; i++)
			for (int j = 0; j < column; j++)
				data[i][j] = m.data[i][j];
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename E>
void matrix<E>::operator= (const matrix<E>& m)
{
    for(int i = 0; i < row; i++)
        delete[] data[i];
    delete[] data;
    row = m.row;
    column = m.column;
    data = new E*[row];
    for (int i = 0; i < row; i++)
        data[i] = new E[column];
    for (int i = 0; i < row; i++)
			for (int j = 0; j < column; j++)
				data[i][j] = m.data[i][j];
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename E>
void matrix<E>::print() const
{
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < column; j++)
			cout << setw(9) << setiosflags(ios::fixed) << setprecision(15) << data[i][j] << '\t';
		cout << endl;
	}
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename E>
char* matrix<E>::ToString() const
{
	string str;
	stringstream omem(str);

	omem << '\t';
	for (int i = 0; i < column; i++)
		omem << setw(9) << i << '\t';
	omem << '\n';

	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < column + 1; j++)
		{
			if (j == 0)
				omem << i << '\t';
			else
				omem << setw(10) << setprecision(4) << setiosflags(ios::fixed) << data[i][j-1] << '\t';
		}
		omem << '\n';
	}
	omem << '\0';

	int SIZE = row * column * 2 * 12 + row + column + 1;
	stringbuf *pbuf = omem.rdbuf();
	char* membuff = new char[SIZE];
	pbuf->sgetn (membuff,SIZE);

	return membuff;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename E>
E& matrix<E>::access(int r, int c)
{
	return data[r][c];
	// TODO: insert return statement here
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename E>
int matrix<E>::Rows() const
{
	return row;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename E>
int matrix<E>::Columns() const
{
	return column;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename E>
matrix<E> matrix<E>::operator() (int r1, int c1, int r2, int c2) const
{
    if (r2 == -1)
        r2 = row - 1;
    if (c2 == -1)
        c2 = column - 1;

    matrix<E> result(r2 - r1 + 1, c2 - c1 + 1);
    int ii = 0;
    int jj = 0;
    for(int i = r1; i < r2 + 1; i++)
    {
        for(int j = c1; j < c2 + 1; j++)
        {
            result.data[ii][jj] = data[i][j];
            jj++;
        }
        ii++;
        jj = 0;
    }
    return result;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename E>
matrix<E> matrix<E>::operator+(const matrix<E> m) const
{
    matrix<E> a(row, column);
	if (row == m.row && column == m.column)
	{
		for (int i = 0; i < row; i++)
			for (int j = 0; j < column; j++)
				a.data[i][j] = data[i][j] + m.data[i][j];
	}
	else if (row == m.row && m.column == 1)
    {
        for (int j = 0; j < column; j++)
            for (int i = 0; i < row; i++)
				a.data[i][j] = data[i][j] + m.data[i][0];
    }
    else if (column == m.column && m.row == 1)
    {
        for (int i = 0; i < row; i++)
            for (int j = 0; j < column; j++)
				a.data[i][j] = data[i][j] + m.data[0][j];
    }
    else if (row == m.row && column == 1)
    {
        matrix<E> b(m.row, m.column);
        for (int j = 0; j < m.column; j++)
            for (int i = 0; i < m.row; i++)
				b.data[i][j] = data[i][0] + m.data[i][j];
        return b;
    }
    else if (column == m.column && row == 1)
    {
        matrix<E> b(m.row, m.column);
        for (int i = 0; i < m.row; i++)
            for (int j = 0; j < m.column; j++)
				b.data[i][j] = data[0][j] + m.data[i][j];
        return b;
    }
	else
		cout << "Not allowed !" << endl;
	return a;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename E>
matrix<E> matrix<E>::operator-(const matrix<E> m) const
{
	matrix<E> a(row, column);
	if (row == m.row && column == m.column)
	{
		for (int i = 0; i < row; i++)
			for (int j = 0; j < column; j++)
				a.data[i][j] = data[i][j] - m.data[i][j];
	}
	else if (row == m.row && m.column == 1)
    {
        for (int j = 0; j < column; j++)
            for (int i = 0; i < row; i++)
				a.data[i][j] = data[i][j] - m.data[i][0];
    }
    else if (column == m.column && m.row == 1)
    {
        for (int i = 0; i < row; i++)
            for (int j = 0; j < column; j++)
				a.data[i][j] = data[i][j] - m.data[0][j];
    }
    else if (row == m.row && column == 1)
    {
        matrix<E> b(m.row, m.column);
        for (int j = 0; j < m.column; j++)
            for (int i = 0; i < m.row; i++)
				b.data[i][j] = data[i][0] - m.data[i][j];
        return b;
    }
    else if (column == m.column && row == 1)
    {
        matrix<E> b(m.row, m.column);
        for (int i = 0; i < m.row; i++)
            for (int j = 0; j < m.column; j++)
				b.data[i][j] = data[0][j] - m.data[i][j];
        return b;
    }
	else
		cout << "Not allowed !" << endl;
	return a;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename E>
matrix<E> matrix<E>::operator/(const matrix<E> m) const
{
	if (column == m.column && row == m.row)
	{
		matrix<E> a(row, column);
		for (int i = 0; i < row; i++)
			for (int j = 0; j < column; j++)
                a.data[i][j] = data[i][j] / m.data[i][j];
		return a;
	}
	else
	{
		cout << "Not allowed !" << endl;
		return matrix<E>(0, 0, 0);
	}
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename E>
matrix<E> matrix<E>::operator*(const matrix<E> m) const
{
	if (column == m.column && row == m.row)
	{
		matrix<E> a(row, column);
		for (int i = 0; i < row; i++)
			for (int j = 0; j < column; j++)
                a.data[i][j] = data[i][j] * m.data[i][j];
		return a;
	}
	else
	{
		cout << "Not allowed !" << endl;
		return matrix<E>(0, 0, 0);
	}
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename E>
matrix<E> matrix<E>::operator+(E n) const
{
	matrix<E> a(row, column, 0);
	for (int i = 0; i < row; i++)
		for (int j = 0; j < column; j++)
			a.data[i][j] = data[i][j] + n;
	return a;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename E>
matrix<E> matrix<E>::operator-(E n) const
{
	matrix<E> a(row, column, 0);
	for (int i = 0; i < row; i++)
		for (int j = 0; j < column; j++)
			a.data[i][j] = data[i][j] - n;
	return a;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename E>
matrix<E> matrix<E>::operator/(E n) const
{
	matrix<E> a(row, column, 0);
	for (int i = 0; i < row; i++)
		for (int j = 0; j < column; j++)
			a.data[i][j] = data[i][j] / n;
	return a;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename E>
matrix<E> matrix<E>::operator*(E n) const
{
	matrix<E> a(row, column, 0);
	for (int i = 0; i < row; i++)
		for (int j = 0; j < column; j++)
			a.data[i][j] = data[i][j] * n;
	return a;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename E>
matrix<E> matrix<E>::dot(const matrix<E> m) const
{
    if (column == m.row)
	{
		matrix<E> a(row, m.column);
		for (int i = 0; i < row; i++)
			for (int j = 0; j < m.column; j++)
				for (int k = 0; k < column; k++)
					a.data[i][j] += data[i][k] * m.data[k][j];
		return a;
	}
	else
	{
		cout << "Not allowed !" << endl;
		return matrix<E>(0, 0, 0);
	}
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename E>
matrix<E> matrix<E>::divide(const matrix<E> divisor) const
{
    return this->dot(divisor.Inverse());
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename E>
bool matrix<E>::operator==(const matrix<E> m) const
{
	if (row == m.row && column == m.column)
	{
		for (int i = 0; i < row; i++)
			for (int j = 0; j < column; j++)
				if (!(data[i][j] == m.data[i][j]))
					return  false;
	}
	else
		return false;
	return true;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename E>
bool matrix<E>::operator!=(const matrix<E> m) const
{
	return !(*(this) == m);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename E>
bool matrix<E>::IsIdentity() const
{
	if (row == column)
	{
		for (int i = 0; i < row; i++)
			for (int j = 0; j < column; j++)
			{
				if (i == j)
				{
					if (data[i][j] != 1)
					{
						return false;
					}
				}
				else
				{
					if (data[i][j] != 0)
						return false;
				}
			}
	}
	else
		return false;
	return true;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename E>
bool matrix<E>::IsIdempotent() const
{
	return (*(this) * *(this) == *(this));
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename E>
bool matrix<E>::IsSquare() const
{
	return (row==column);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename E>
bool matrix<E>::IsSymmetric() const
{
	if (column == row)
	{
		for (int i = 0; i < row; i++)
			for (int j = 0; j < column; j++)
				if (i != j)
				{
					if (data[i][j] != data[j][i])
						return false;
				}
	}

	else
		return false;
	return true;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename E>
bool matrix<E>::IsUpperTriangle() const
{
	if (row == column)
	{
		for (int i = 0; i < row; i++)
			for (int j = 0; j < column; j++)
				if (i > j)
				{
					if (data[i][j] != 0)
						return false;
				}
	}
	else
		return false;
	return true;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename E>
bool matrix<E>::IsLowerTriangle() const
{
	if (row == column)
	{
		for (int i = 0; i < row; i++)
			for (int j = 0; j < column; j++)
				if (i < j)
				{
					if (data[i][j] != 0)
						return false;
				}
	}
	else
		return false;
	return true;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename E>
matrix<E> matrix<E>::Inverse() const
{
	/* Augmenting identity matrix into A */
	matrix<E> Inv(row, column * 2);
	for (int i = 0; i < row; i++)
		for (int j = 0; j < column; j++)
			Inv.access(i, j) = data[i][j];

	for (int i = 0; i < row; i++)
		for (int j = column; j < column * 2; j++)
		{
			if (i == j - column)
				Inv.access(i, j) = 1;
			else
				Inv.access(i, j) = 0;
		}

	/* Gaussian elimination */
	for (int i = 0; i < row; i++)
	{
		float divisor = Inv.access(i, i);
		for (int k = 0; k < column * 2; k++)
			Inv.access(i, k) /= divisor;

		for (int j = 0; j < column; j++)
		{
			if (i == j)
				continue;
			float mult = Inv.access(j, i);
			for (int k = 0; k < column * 2; k++)
				Inv.access(j, k) -= mult * Inv.access(i, k);

		}
	}


	return Inv(0, column);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename E>
matrix<E> matrix<E>::CholeskyInverse() const
{
	matrix<E> LT = this->LowerTri();
	matrix<E> LTinv = LT.LTinverse();
	matrix<E> LtinvTrans = LTinv.transpose();
	return LTinv.dot(LtinvTrans);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename E>
matrix<E> matrix<E>::SlowInverse() const
{
    matrix<E> m(row, column);
	E DetDiv = this->determinant();
	if (DetDiv != 0)
	{
        matrix<E> adj(row, column);
		for (int i = 0; i < row; i++)
			for (int j = 0; j < column; j++)
			{
				matrix<E> temp(row - 1, column - 1);
				int tempi = 0;
				int tempj = 0;
				for (int k = 0; k < row; k++)
					for (int l = 0; l < column; l++)
						if (k != i && l != j)
						{
							temp.data[tempi][tempj] = data[k][l];
							tempj++;
							if (tempj == column-1)
							{
								tempi++;
								tempj = 0;
							}
						}
				adj.data[i][j] = temp.determinant();
				if (((i + j) % 2) == 1)
					adj.data[i][j] *= -1;
			}
		matrix<E> invDiv(row, column);
		invDiv = adj * (1 / DetDiv);
		invDiv = invDiv.transpose();
		return invDiv;
	}
	else
    {
        cout << "Can't calculate inverse because the determinant is zero!" << endl;
		return m;
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename E>
matrix<E> matrix<E>::transpose() const
{
	matrix<E> a(column, row, 0);
	for (int i = 0; i < column; i++)
		for (int j = 0; j < row; j++)
			a.data[i][j] = data[j][i];
	return a;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename E>
matrix<E> matrix<E>::sum(string choice) const
{
    if (choice == "row")
    {
        matrix<E> result(1,column);
        for (int i = 0; i < column; i++)
            for (int j = 0; j < row; j++)
                result.data[0][i] += data[j][i];
        return result;
    }
    else if (choice == "column")
    {
        matrix<E> result(row,1);
        for (int i = 0; i < row; i++)
            for (int j = 0; j < column; j++)
                result.data[i][0] += data[i][j];
        return result;
    }
    else
        cout << "Wrong argument!" <<endl;
    return *this;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename E>
E matrix<E>::sumall() const
{
    E result = 0;
    for (int i = 0; i < row; i++)
        for (int j = 0; j < column; j++)
            result += data[i][j];
    return result;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename E>
E matrix<E>::MaxElement() const
{
	float maxelem = -1.2E-38;
	for (int i = 0; i < row; i++)
        for (int j = 0; j < column; j++)
            if (data[i][j] > maxelem)
                maxelem = data[i][j];
    return maxelem;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename E>
E matrix<E>::MinElement() const
{
	float minelem = 1.2E38;
	for (int i = 0; i < row; i++)
        for (int j = 0; j < column; j++)
            if (data[i][j] < minelem)
                minelem = data[i][j];
    return minelem;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename E>
E matrix<E>::determinant() const
{
	matrix<E> temp(row, column);
	for (int i = 0; i < row; i++)
		for (int j = 0; j < column; j++)
			temp.data[i][j] = data[i][j];

	if (temp.IsSquare())
	{
		E* temprow = new E(row);
		for (int i = 0; i < row-1; i++)
			for (int j = i + 1; j < row; j++)
			{
				if (temp.data[i][i] == 0)
					for (int k = i + 1; k < row; k++)
						if (temp.data[k][i] != 0)
						{
							for (int l = 0; l < row; l++)
							{
								temprow[l] = temp.data[i][l];
								temp.data[i][l] = temp.data[k][l];
								temp.data[k][l] = temprow[l];
							}
							break;
						}
				E pivot = temp.data[j][i] / temp.data[i][i];
				for (int k = i; k < column; k++)
					temp.data[j][k] -= temp.data[i][k] * pivot;
			}

	}
	else
	{
		cout << "\nThe matrix is not square!" << endl;
		return 0;
	}

	E det = 1;
	for (int i = 0; i < row; i++)
		det *= temp.data[i][i];
	return det;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename E>
void matrix<E>::Fill(E e)
{
	for (int i = 0; i < row; i++)
		for (int j = 0; j < column; j++)
			data[i][j] = e;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename E>
matrix<E> matrix<E>::LowerTri() const
{
	matrix<E> result(row, column);
	result = *this;
	for(int i = row - 1; i >= 1; i--)
        for(int j = i - 1; j >= 0; j--)
        {
			/* Zeros the upper rows */
            E frac = result.access(j, i) / result.access(i, i);
			for (int k = column - 1; k >= 0; k--)
				result.access(j, k) = result.access(j, k) - result.access(i, k) * frac;

			/* Checks if there is a zero in diagonal */
			if (result.access(i - 1, i - 1) == 0)
			{
				for (int k = 2; k <= i; k++)
				{
					if (result.access(i - k, i - 1) != 0)
					{
						for (int ii = 0; ii < column; ii++)
						{
							E temp;
							temp = result.access(i - 1, ii);
							result.access(i - 1, ii) = result.access(i - k, ii);
							result.access(i - k, ii) = temp;
						}
						j++;
						break;
					}
					/* checks if there are no more non-zero elements above diagonal */
					if (k == i)
					{
						cout << "Matrix is singular!" << endl;
						return result;
					}
				}
			}
		}

    return result;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename E>
matrix<E> matrix<E>::LTinverse() const
{
	matrix<E> result(row, column);
	matrix<E> C(row, column);
	matrix<E> diag(row, column);

	/* Making diagonal = 1 */
	for (int i = 0; i < row; i++)
	{
		diag.access(i, i) = 1.0 / data[i][i];
		result.access(i, i) = 1;
	}
	C = diag.dot(*this);

	/* Step 1 */
	for (int i = 0; i < row - 1; i++)
		result.access(i + 1, i) = -1 * C.access(i + 1, i);

	/* Step 2 */
	for (int i = 0; i < row - 2; i++)
		result.access(i + 2, i) = C.access(i + 1, i) * C.access(i + 2, i + 1) - C.access(i + 2, i);

	/* Step 3 */
	for (int i = 3; i < row; i++)
	{
		for (int j = 0; j < row - i; j++)
		{
			for (int q = j + 1; q <= i + j; q++)
			{
				result.access(i + j, j) += result.access(i + j, q) * C.access(q, j);
			}
			result.access(i + j, j) *= -1;
		}
	}

	/* Step 4 */
	return result.dot(diag);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename E>
matrix<E> matrix<E>::getlog() const
{
	matrix<E> a(row, column, 0);
	for (int i = 0; i < row; i++)
		for (int j = 0; j < column; j++)
			a.data[i][j] = log(data[i][j]);
	return a;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename E>
matrix<E> matrix<E>::square() const
{
	matrix<E> a(row, column, 0);
	for (int i = 0; i < row; i++)
		for (int j = 0; j < column; j++)
			a.data[i][j] = data[i][j] * data[i][j];
	return a;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename E>
matrix<E> matrix<E>::Sqrt() const
{
	matrix<E> a(row, column, 0);
	for (int i = 0; i < row; i++)
		for (int j = 0; j < column; j++)
			a.data[i][j] = sqrt(data[i][j]);
	return a;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename E>
matrix<E>::~matrix()
{
    for(int i = 0; i < row; i++)
        delete[] data[i];
    delete[] data;
}
#endif // !matrix_head
