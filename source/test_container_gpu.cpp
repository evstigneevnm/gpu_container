#include<container_gpu.hpp>
#include<iostream>
#include<algorithm>
#include<iterator>


int main(int argc, char const *argv[])
{
    int N = 277440;
    int steps = 2000;
    int pci_id = 4;
    if(init_cuda(pci_id) == 0)
    {
        std::cout << "failed to initialize cuda device" << std::endl;
        return 0;
    }
    container_gpu<double> C;
    C.init(N);
    double *ccc = new double[N];
    double *ccc_d = device_allocate<double>(N);



    for(int j = 0;j<steps;j++)
    {
        std::fill(ccc, ccc+N, j);
        host_2_device_cpy(ccc_d, ccc, N);
        C.push_back(ccc_d);
        std::cout << "value = " << ccc[0] << " size = " << C.get_size() << " capacity = " << C.capacity() << std::endl;
    }
    C.shrink_to_fit();
    std::cout << "size = " << C.get_size() << " capacity = " << C.capacity() << std::endl;
    for(int j=0;j<steps;j++)
    {
        double val;
        device_2_host_cpy(&val, &C.data()[j*N+N-1], 1);
        std::cout << j << " " << val << std::endl;
    }

    CUDA_SAFE_CALL(cudaFree(ccc_d));
    delete [] ccc;

    return 0;
}