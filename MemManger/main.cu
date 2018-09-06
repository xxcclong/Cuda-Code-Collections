#include <bfc_mmanager.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
struct timeval b, e;

int main()
{
    int n = 100;
    std::size_t initMem = 500 << 20;//52428800;
    printf("%zu", initMem);
    std::size_t maxMem = 10000;
    if(n * maxMem > initMem)
    {
        printf("error, may exceed mem limit, return -1\n");
        return -1;
    }
    BFCMemManager *m = new BFCMemManager(initMem);
    //srand((unsigned)time(0));
    std::vector<std::size_t> use_mem;
    std::vector<std::size_t> use2_mem;
    std::vector<Storage*> my_mems;
    std::vector<void*> sys_mems;
    std::vector<bool> ifFree;
    for(int i = 0;i<n;++i)
    {
        size_t s = (std::size_t)(rand() % maxMem);
        use_mem.push_back(s);
        size_t s2 = (std::size_t)(rand() % maxMem);
        use2_mem.push_back(s2);
        if(rand() % 2 == 1) ifFree.push_back(true);
        else ifFree.push_back(false);
    }
    // alloc1
    printf("start alloc 1\n");
    gettimeofday(&b, NULL);
    //std::size_t all = 0;
    for(int i = 0;i<use_mem.size();++i)
    {
        //all += use_mem[i];
        //printf("%zu\n", all);
        my_mems.push_back(m->alloc(use_mem[i]));
    }
#ifdef CUDA
    cudaDeviceSynchronize();
#endif
    gettimeofday(&e, NULL);
    double parallel_time = (double)(e.tv_sec-b.tv_sec + 1e-6 * e.tv_usec - 1e-6*b.tv_usec);
    printf("my mem manager use time %lf\n", parallel_time);

    gettimeofday(&b, NULL);
    for(int i = 0;i<use_mem.size();++i)
    {
        void* p = NULL;
#ifdef CUDA
        cudaMalloc(&p, use_mem[i]);
#else
        p = std::malloc(use_mem[i]);
#endif
        sys_mems.push_back(p);
    }
#ifdef CUDA
    cudaDeviceSynchronize();
#endif
    gettimeofday(&e, NULL);
    double parallel_time2 = (double)(e.tv_sec-b.tv_sec + 1e-6 * e.tv_usec - 1e-6*b.tv_usec);
    printf("sys mem manager use time %lf\n", parallel_time2);
    printf("acculerate ratio %lf\n", parallel_time2 / parallel_time);

    //m->show();
    // free1
    printf("start free 1\n");
    gettimeofday(&b, NULL);
    for(int i = 0;i<use_mem.size();++i)
    {
        //my_mems.push_back( m->alloc(use_mem[i]));
        if(ifFree[i]) 
        {
            m->free(*(my_mems[i]));
            my_mems[i] = NULL;
            use_mem[i] = 0;
        }

    }
#ifdef CUDA
    cudaDeviceSynchronize();
#endif
    gettimeofday(&e, NULL);
    parallel_time = (double)(e.tv_sec-b.tv_sec + 1e-6 * e.tv_usec - 1e-6*b.tv_usec);
    printf("my mem manager use time %lf\n", parallel_time);

    gettimeofday(&b, NULL);
    for(int i = 0;i<use_mem.size();++i)
    {
        if(ifFree[i])
        {
#ifdef CUDA
        cudaFree(sys_mems[i]);
#else
        std::free(sys_mems[i]);
#endif
        sys_mems[i] = NULL;
        }
    }
#ifdef CUDA
    cudaDeviceSynchronize();
#endif
    gettimeofday(&e, NULL);
    parallel_time2 = (double)(e.tv_sec-b.tv_sec + 1e-6 * e.tv_usec - 1e-6*b.tv_usec);
    printf("sys mem manager use time %lf\n", parallel_time2);
    printf("acculerate ratio %lf\n", parallel_time2 / parallel_time);
    //m->show();

    // alloc2
    printf("start alloc 2\n");
    gettimeofday(&b, NULL);
    for(int i = 0;i<use2_mem.size();++i)
    {
        my_mems.push_back( m->alloc(use2_mem[i]));
    }
#ifdef CUDA
    cudaDeviceSynchronize();
#endif
    gettimeofday(&e, NULL);
    parallel_time = (double)(e.tv_sec-b.tv_sec + 1e-6 * e.tv_usec - 1e-6*b.tv_usec);
    printf("my mem manager use time %lf\n", parallel_time);

    gettimeofday(&b, NULL);
    for(int i = 0;i<use2_mem.size();++i)
    {
        void* p = NULL;
#ifdef CUDA
        cudaMalloc(&p, use2_mem[i]);
#else 
        p = std::malloc(use2_mem[i]);
#endif
        sys_mems.push_back(p);
    }
#ifdef CUDA
    cudaDeviceSynchronize();
#endif
    gettimeofday(&e, NULL);
    parallel_time2 = (double)(e.tv_sec-b.tv_sec + 1e-6 * e.tv_usec - 1e-6*b.tv_usec);
    printf("sys mem manager use time %lf\n", parallel_time2);
    printf("acculerate ratio %lf\n", parallel_time2 / parallel_time);
    //m->show();
    
    // free2
    printf("start free  2\n");
    // check
    if(my_mems.size() != sys_mems.size())
    {
        printf("diff size\n");
        return -2;
    }
    for(int i = 0;i<my_mems.size();++i)
    {
        if(my_mems[i] == NULL && sys_mems[i]!=NULL)
        {
            printf("diff null\n");
            return -3;
        }
        if(my_mems[i] != NULL && sys_mems[i] == NULL)
        {
            printf("diff null\n");
            return -3;
        }
    }
    gettimeofday(&b, NULL);
    for(int i = 0;i<my_mems.size();++i)
    {
        //my_mems.push_back( m->alloc(use_mem[i]));
        if(my_mems[i] == NULL)
            continue;
        m->free(*(my_mems[i]));
    }
#ifdef CUDA
    cudaDeviceSynchronize();
#endif
    gettimeofday(&e, NULL);
    parallel_time = (double)(e.tv_sec-b.tv_sec + 1e-6 * e.tv_usec - 1e-6*b.tv_usec);
    printf("my mem manager use time %lf\n", parallel_time);

    gettimeofday(&b, NULL);
    for(int i = 0;i<sys_mems.size();++i)
    {
        if(sys_mems[i] == NULL) continue;
#ifdef CUDA
        cudaFree(sys_mems[i]);
#else 
        std::free(sys_mems[i]);
#endif
    }
#ifdef CUDA
    cudaDeviceSynchronize();
#endif
    gettimeofday(&e, NULL);
    parallel_time2 = (double)(e.tv_sec-b.tv_sec + 1e-6 * e.tv_usec - 1e-6*b.tv_usec);
    printf("sys mem manager use time %lf\n", parallel_time2);
    printf("acculerate ratio %lf\n", parallel_time2 / parallel_time);

    //m->show();

    
    delete m;
    return 0;
}
