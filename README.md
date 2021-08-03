# gpu_container
A simple class that implements an adaptive container for a GPU storage using CUDA.
Can be used analogous to the std::vector<type>, contains the following methods:
- init(size of a single data type)
- push_back(type val_)
- shrink_to_fit()
- get_size()
- capacity()
- type data()  <- access to internal part of the container.
                  
Constructor takes two optional parameters:
- container(int muliplyer_ = 2, int init_steps_ = 100)
muliplyer_ - adjusts the internal storage if the data doesnot fit for the next push_back()
init_steps_ - amount of data that is initially reserved in the internal storage.             
