#include<container.hpp>
#include<iostream>
#include<algorithm>
#include<iterator>


int main(int argc, char const *argv[])
{
    int N = 100000;
    int steps = 764;
    container<double> C;
    C.init(N);
    double *ccc = new double[N];



    for(int j = 0;j<steps;j++)
    {
        std::fill(ccc, ccc+N, j);
        C.push_back(ccc);
        std::cout << "value = " << ccc[0] << " size = " << C.get_size() << " capacity = " << C.capacity() << std::endl;
    }

    C.shrink_to_fit();
    std::cout << "size = " << C.get_size() << " capacity = " << C.capacity() << std::endl;
    for(int j=0;j<steps;j++)
    {
        std::cout << j << " " << C.data()[j*N+N-1] << std::endl;
    }

    delete [] ccc;

    return 0;
}