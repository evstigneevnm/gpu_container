#ifndef __DYNAMIC_INDEX_TABLE_GPU_H__
#define __DYNAMIC_INDEX_TABLE_GPU_H__

#include <scfd/utils/device_tag.h>
#include <scfd/utils/cuda_safe_call.h>

#include <scfd/memory/cuda.h>
#include <scfd/arrays/tensorN_array.h>
#include <scfd/arrays/tensorN_array_nd.h>
#include <scfd/arrays/last_index_fast_arranger.h>

namespace detail{

template<class Ord>
class dynamic_index_table_gpu
{
public:

    using gpu_container_2_t = scfd::arrays::tensor0_array_nd< Ord, 2, scfd::memory::cuda_device >;
    using gpu_container_2_view_t = typename gpu_container_2_t::view_type;

    using gpu_container_1_t = scfd::arrays::tensor0_array_nd< Ord, 1, scfd::memory::cuda_device >;
    using gpu_container_1_view_t = typename gpu_container_1_t::view_type;

    using gpu_bool_1_t = scfd::arrays::tensor0_array_nd< bool, 1, scfd::memory::cuda_device >;
    using gpu_bool_1_view_t = typename gpu_bool_1_t::view_type;


    dynamic_index_table_gpu(unsigned char verbose_level_ = 1):
    verbose_level(verbose_level_)
    {

    }
    ~dynamic_index_table_gpu()
    {
        __delete();
    }

    void init(Ord N_rows_, Ord N_cols_ = 10)//host function, N_cols initial
    {
        gpu.N_rows = N_rows_;
        gpu.N_cols = N_cols_;
        N_cols_base = N_cols_;
        __allocate();
    }

    bool out_of_memory() //returns false if the last call used enough memory and no overflow occured
    {
        bool res_;
        gpu_bool_1_view_t out_of_memory_view(gpu.out_of_memory);
        res_ = out_of_memory_view(0);
        out_of_memory_view.release(false);
        return res_;

    }
    void adjust() //called if insuffcient memory is used.
    {
        __delete();
        gpu.N_cols += N_cols_base;
        if(verbose_level>=1)
        {
            std::cout << "dynamic_index_table_gpu:: adjust(): N_rows = " << gpu.N_rows << " N_cols = " << gpu.N_cols << std::endl;
        }
        // add check if free device memory is insufficient
        
        auto device_free_mem = __get_free_current_device_memory();
        auto mem_needed = (size_t(gpu.N_cols)*size_t(gpu.N_rows) + size_t(gpu.N_rows))*sizeof(Ord) + sizeof(gpu_t);
        
        if(verbose_level>=2)
        {
            std::cout << "dynamic_index_table_gpu:: adjust(): device free mem = " << int(device_free_mem/1.0e6) << "MB, required memory = " << int(mem_needed/1.0e6) << "MB, free memory after allocation = " << int((device_free_mem - mem_needed)/1.0e6) << "MB." << std::endl;
        }
        if(device_free_mem <= mem_needed)
        {
            throw std::runtime_error("dynamic_index_table_gpu:: adjust(): insufficient device memory.");
        }
        
        __allocate();
    }

    void plot_debug()
    {
        gpu_container_2_view_t container_all_view(gpu.container_all);

        for(int j = 0; j<gpu.N_rows;j++)
        {
            for(int k = 0;k<gpu.N_cols;k++)
            {
                std::cout << container_all_view(j,k) << " ";
            }
            std::cout << std::endl;
        }


    }

    struct gpu_t
    {
        __DEVICE_TAG__ void add(Ord j, Ord k)
        {

            if(!out_of_memory(0) )
            {
                if( !check(j, k) )
                {
                    auto id_ = atomicAdd(&container_num[j], 1);
                    if(id_ >= N_cols )
                    {
                        out_of_memory(0) = true;
                    }
                    else
                    {
                        container_all(j, id_) = k;
                    }
                }
            }
        }

        __DEVICE_TAG__ bool check(Ord j, Ord k)
        {
            bool found = false;
            if(!out_of_memory(0))
            {
                auto n_ids = container_num[j];
                for(int l = 0; l < n_ids; l++)
                {
                    if( container_all(j, l) == k )
                        found = true;
                }
            }
            return found;
        }


        
        gpu_bool_1_t out_of_memory;

        gpu_container_2_t container_all;
        Ord* container_num = nullptr;
        Ord N_cols;
        Ord N_rows;


    };
    gpu_t gpu;



private:
    unsigned char verbose_level = 1;
    Ord N_cols_base;

    void __delete()
    {
        gpu.container_all.free();
        gpu.out_of_memory.free();

        if(gpu.container_num != nullptr)
        {
            cudaFree(gpu.container_num);
            gpu.container_num = nullptr;
        }

    }

    void __allocate()
    {

        gpu.out_of_memory.init(1);
        gpu.container_all.init(gpu.N_rows, gpu.N_cols);
        CUDA_SAFE_CALL(cudaMalloc( &gpu.container_num, gpu.N_cols*sizeof(Ord) ) );   
        
        gpu_bool_1_view_t out_of_memory_view(gpu.out_of_memory);
        out_of_memory_view(0) = false;
        out_of_memory_view.release(true);
    }

    size_t __get_free_current_device_memory()
    {
        size_t free_device_mem;
        size_t total_device_mem;
        CUDA_SAFE_CALL(cudaMemGetInfo(&free_device_mem, &total_device_mem)) ;
        return free_device_mem;
    }



};


}


#endif