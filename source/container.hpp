#ifndef __CONTAINER_HPP__
#define __CONTAINER_HPP__


/* container that miniques the behaviour of a CUDA-based container */

#include<cstdlib>
#include<cstring>
#include<stdexcept>

template<class T>
class container
{
public:
    container(int muliplyer_ = 2, int init_steps_ = 100):
    muliplyer(muliplyer_),
    steps(init_steps_)
    {
    }
    ~container()
    {
        if(storage != nullptr)
        {
            std::free(storage);
        }

    }


    void init(int size_)
    {
        size = size_;
        storage = (T*)std::malloc(size*steps*sizeof(T));
        init_done = true;
    }


    void push_back(T* val_)
    {
        if(!init_done)
        {
            throw std::logic_error("calling push_back withough initialization. Call init(size).");
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

    int get_size()
    {
        return steps_pushed;
    }
    int capacity()
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
    int steps;
    int size;
    int steps_pushed = 0;
    int muliplyer = 2;

    void _realloc()
    {
        T* __storage = nullptr;
        int steps_new = steps*muliplyer;
        __storage = (T*)std::malloc(size*steps_new*sizeof(T));
        if(__storage == nullptr)
        {
            throw std::runtime_error("failed to allocate " + std::to_string(size*steps_new*sizeof(T)) + "bytes of RAM");
        }
        std::memcpy( static_cast<T*>(&__storage[0]), static_cast<T*>(storage), steps*size*sizeof(T) );
        steps = steps_new;
        std::free(storage);
        storage = __storage;

    }

    void _fit()
    {
        T* __storage = nullptr;
        __storage = (T*)std::malloc(size*steps_pushed*sizeof(T));
        if(__storage == nullptr)
        {
            throw std::runtime_error("failed to allocate " + std::to_string(size*steps_pushed*sizeof(T)) + "bytes of RAM");
        }
        std::memcpy( static_cast<T*>(&__storage[0]), static_cast<T*>(storage), steps_pushed*size*sizeof(T) );    
        
        steps = steps_pushed;
        std::free(storage);
        storage = __storage;

    }

    void inline _add(T* val_)
    {
            std::memcpy( static_cast<T*>(&storage[size*steps_pushed]), static_cast<T*>(val_), size*sizeof(T) );
            steps_pushed++;        
    }

};



#endif