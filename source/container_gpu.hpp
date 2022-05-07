#ifndef __CONTAINER_GPU_HPP__
#define __CONTAINER_GPU_HPP__


/* container_gpu that miniques the behaviour of a CUDA-based container_gpu */

#include<cstdlib>
#include<cstring>
#include<stdexcept>
#include<utils/cuda_support.h>
#include<utils/cuda_safe_call.h>

template<class T>
class container_gpu
{
public:
    container_gpu(size_t muliplyer_ = 2, size_t init_steps_ = 100):
    muliplyer(muliplyer_),
    steps(init_steps_)
    {
    }
    ~container_gpu()
    {
        if(storage != nullptr)
        {
            cudaFree(storage);
        }
    }


    void init(size_t size_)
    {
        size = size_;
        storage = device_allocate<T>(size*steps);
        init_done = true;
    }


    void push_back(T* val_)
    {
        if(!init_done)
        {
            throw std::logic_error("calling push_back() withough initialization. Call init(size) first.");
        }
        if(steps_pushed<steps)
        {
            _add(val_);
        }
        else
        {
            _realloc();
            _add(val_);
        }
    }

    void shrink_to_fit()
    {
        _fit();
    }

    size_t get_size()
    {
        return steps_pushed;
    }
    size_t capacity()
    {
        return steps;
    }

    T* data()
    {
        return storage;
    }

private: 
    T* storage = nullptr;
    bool init_done = false;
    size_t steps;
    size_t size;
    size_t steps_pushed = 0;
    size_t muliplyer = 2;

    void _realloc()
    {
        T* __storage = nullptr;
        size_t steps_new = steps*muliplyer;
        try
        {
            __storage = (T*)device_allocate<T>(size*steps_new);
        }
        catch(const std::runtime_error& e)
        {
            throw std::runtime_error("_realloc: failed to allocate " + std::to_string(1.0*size*steps_new*sizeof(T)/1024.0/1024.0/1024.0) + " Gb of RAM");
        }
        device_2_device_cpy<T>(storage, &__storage[0], steps*size);
        steps = steps_new;
        CUDA_SAFE_CALL(cudaFree(storage));
        storage = __storage;
    }

    void _fit()
    {
        T* __storage = nullptr;
        try
        {        
            __storage = (T*)device_allocate<T>(size*steps_pushed);
        }
        catch(const std::runtime_error& e)
        {
            throw std::runtime_error("_fit: failed to allocate " + std::to_string(1.0*size*steps_pushed*sizeof(T)/1024.0/1024.0/1024.0) + " Gb of RAM");
        }
        device_2_device_cpy<T>(storage, &__storage[0], steps_pushed*size);
        steps = steps_pushed;
        CUDA_SAFE_CALL(cudaFree(storage));
        storage = __storage;
    }

    void inline _add(T* val_)
    {
        try
        {             
            device_2_device_cpy<T>(val_, &storage[size*steps_pushed], size);
        }
        catch(const std::runtime_error& e)
        {
            throw std::runtime_error("_add: failed to allocate " + std::to_string(1.0*size*steps_pushed*sizeof(T)/1024.0/1024.0/1024.0) + " Gb of RAM");            
        } 
        steps_pushed++;        
    }

};



#endif