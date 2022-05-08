/******************************************************************************

                              Online C++ Compiler.
               Code, Compile, Run and Debug C++ program online.
Write your code in this editor and press "Run" button to compile and execute it.

*******************************************************************************/

#include <iostream>
#include <cmath>

template<class T>
struct address_link
{
    T* data = nullptr;
    size_t data_counter = 0;
    
    address_link* link_next = nullptr;
    address_link* link_previous = nullptr;
    size_t link_number = 0;
    
    ~address_link()
    {
        std::cout << "~: link_number = " << link_number << std::endl; 
        if(link_next!=nullptr)
            delete link_next;
            
        if(data != nullptr)
        {
            free(data);
            data = nullptr;
        }   
        std::cout << "~: number " << link_number << ": i'm free =)" << std::endl;
        
    }
};

template<class Ord>
class list
{
public:    
    size_t N;
    using chain_t = address_link<Ord>;
    
    chain_t* link_p;
    
    list(size_t N_):
    N(N_)
    {
        link_p = new chain_t();
        link_p->data = (Ord*)malloc( sizeof(Ord)*N );
        link_p->link_number = 0;
    }
    
    ~list()
    {
        auto current_link = link_p;
        auto prev_link = link_p;
        while(prev_link != nullptr)
        {
            prev_link = current_link->link_previous;
            if(prev_link != nullptr)
            {
                current_link = prev_link;
            }
        }
        link_p = current_link;
        delete link_p;
    }
   
   
    void add(Ord k)
    {
        if(link_p->data_counter>N-1)
        {
            add_link();
        }
        link_p->data[link_p->data_counter++] = k;
    }
    
    Ord get(Ord k)
    {
        size_t which_link = std::floor(k/N);
        if(which_link != link_p->link_number)
        {
            find_link_number(which_link);
        }
        return link_p->data[k - which_link*N];
    }
    
    
private:

    void find_link_number(int which_link)
    {
        if( which_link > link_p->link_number)
        {
            while((which_link != link_p->link_number)&&(link_p->link_next != nullptr))
            {
                auto link_p_l = link_p->link_next;
                if(link_p_l!=nullptr)
                {
                    link_p = link_p_l;
                }
            }
        }
        else if(which_link < link_p->link_number)
        {
            while( (which_link != link_p->link_number)&&(link_p->link_previous != nullptr) )
            {
                auto link_p_l = link_p->link_previous;
                if(link_p_l!=nullptr)
                {
                    link_p = link_p_l;
                }
            }            
        }
        
    }

    void add_link()
    {
        auto current_link = link_p;
        auto next_link = link_p;
        while(next_link != nullptr)
        {
            next_link = current_link->link_next;
            if(next_link != nullptr)
            {
                current_link = next_link;
            }
        }

        auto new_link = new chain_t();
        current_link->link_next = new_link;
        new_link->link_previous = current_link;
        new_link->link_number = current_link->link_number + 1;
        
        new_link->data = (Ord*)malloc(sizeof(Ord)*N);
        link_p = new_link;
        std::cout << current_link << std::endl;
        std::cout << new_link << std::endl << "-------------" << std::endl;
        
    }
    
};

int main()
{
    
    
    list<int> list(4);
    
    for(int j = 0;j<300;j++)
        list.add(j-149);

    for(int j = 0;j<300;j++)
        std::cout << list.get(j) << " ";

    return 0;
}
