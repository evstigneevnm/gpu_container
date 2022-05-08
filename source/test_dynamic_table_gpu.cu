#include <scfd/utils/init_cuda.h>
#include <scfd/utils/cuda_timer_event.h>
#include <scfd/utils/device_tag.h>
#include <dynamic_index_table_gpu.h>
#include <scfd/utils/cuda_timer_event.h>

namespace detail{
namespace kernel{

template<class Ord, class Container, unsigned int MaxN>
__global__ void set_colors(Ord N_cold, Ord N_rows, Container cnt)
{
    Ord idx=blockDim.x * blockIdx.x + threadIdx.x;

    if(idx%10 == 0)
    {
        for(int k = 0; k < MaxN; k++)
        {
            cnt.add(idx, k*100);
        }
    }

}


}
}





int main(int argc, char const *argv[])
{
    
    using gpu_timer_event_t = scfd::utils::cuda_timer_event;

    size_t max_colors = 2000000;
    size_t max_neighbours = 10;

    scfd::utils::init_cuda(-1, 0);
    const unsigned int blocksize = 512;
    
    dim3 dimBlock( blocksize, 1, 1 );
    int blocks_x= std::ceil(1.0*max_colors/( 1.0*blocksize ));
    dim3 dimGrid( blocks_x, 1, 1);

    using ditg_t = detail::dynamic_index_table_gpu<int>;
    ditg_t ditg(2);
    ditg.init(max_colors, max_neighbours);

    bool out_of_memory = true;
    
    gpu_timer_event_t t1, t2;
    t1.record(); 

    while(out_of_memory)
    {
        
        detail::kernel::set_colors<int, ditg_t::gpu_t, 500 ><<<dimGrid, dimBlock>>>(max_colors, max_neighbours, ditg.gpu);
        
        out_of_memory = ditg.out_of_memory();
        if(out_of_memory)
        {
            ditg.adjust();
        }
    }

    t2.record();
    auto execution_time = t2.elapsed_time(t1);
    std::cout << "excution time = " << execution_time << "ms." << std::endl;    

    if(max_colors < 100)
        ditg.plot_debug();

    return 0;
}