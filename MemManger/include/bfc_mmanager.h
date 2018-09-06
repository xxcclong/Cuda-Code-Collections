#ifndef GARLIC_BFC_MMANAGER_H
#define GARLIC_BFC_MMANAGER_H

#include <queue>
#include <set>
#include <iostream>
#include <vector>
#include <unordered_set>
#include <cuda_runtime.h>
//#define DEBUG
#define CUDA
/**
 * @brief BFC Memory Manager
 *
 * BFC Memory manager implementing BFC(best fit with with coalescing algorithm)
 * using bucket sort to find the free chunk in the bins
 * every time a request come in, first find the least chunk whose size >= requested size
 * if found, decide whether split the chunk, if split, the allocated chunk size is exactly the rounded req size
 * else if not found, extend the total memory, and return assign the newly allocated memory to the req
 * finally, construct the Storage with chunk's ptr and chunkId.
 * when freeing the Storage, find the chunk using the chunkId stored and insert the free chunk back to the freeChunkSet
 *
 */
class Storage {
public:
    void * ptr;
    std::size_t nBytes;
    int whichC;
    Storage(void* p, std::size_t nb, int w): ptr(p), nBytes(nb), whichC(w) {}
};
class BFCMemManager {
private:
    int allocTimes_ = 0;
    static const std::size_t kMinAllocationBits = 8;
    static const std::size_t kMinAllocationSize = 1 << kMinAllocationBits;
    /**
     * @brief Memory Chunk
     *
     * chunk is assigned to a certain area of memory
     */
    struct Chunk {
        int allocId = -1;
        int chunkId = -1;
        void* ptr = nullptr;
        std::size_t size = 0;
        std::size_t requestedSize = 0;
        int physicalNext = -1;// prev on the memory
        int physicalPrev = -1;
        int whichBin = -1;// which bin am i in
        bool inUse() { return allocId != -1; }
        Chunk(void* p = nullptr, std::size_t s = 0, std::size_t rs = 0) : ptr(p), size(s), requestedSize(rs) {}
    };

    struct Bin {
        struct ChunkComparator {
            explicit ChunkComparator() {}
            bool operator()(const Chunk* hx, const Chunk* hy) const {
                if (hx->size != hy->size)
                    return hx->size < hy->size;
                return hx->ptr < hy->ptr;
            }
        };
        typedef std::set<Chunk*, ChunkComparator> freeChunkSet;
        freeChunkSet freeChunks;
        std::size_t binSize = 0;
        Bin(std::size_t bs = 0)
            : freeChunks(ChunkComparator()), binSize(bs) {};
        void removeIter(const BFCMemManager::Bin::freeChunkSet::iterator& citer) {
            freeChunks.erase(citer);
        }
        void removeChunkId(Chunk* c) {
            freeChunks.erase(c);
        }
    };

    std::vector<Bin> bins;
    std::vector<Chunk> allChunks;
    std::queue<int> freeChunks;
    int maxChunkNum_ = 10000;
    std::size_t availableMemory = 0;
    std::size_t allMemory_ = 0;
    int binNum_;
    std::vector<void*> allocatedMemoryPtrs;
    std::vector<size_t> allocatedMemoryBytes;
public:
    /**
     * @brief Construct BFC Memory Manager
     *
     * @param device : the device's ptr which the mmanager belong to
     * @param avalMem : total memory can be allocated
     * @param allMem : initial memory owned by the mmanager
     */
    BFCMemManager(std::size_t allMem = 10000);
    ~BFCMemManager();
    /**
     * @brief alloc nBytes memory
     *
     * @param nBytes : how many bytes want to alloc
     * @param hint : to indicate the memory alloc mode (now ignoring)
     * @return Storage containing the ptr pointing at memory space
     */
    Storage* alloc(std::size_t nBytes, unsigned int hint = 0);
    /**
     * free the storage, give the space allocated back to the mmanager
     * @param storage : the storage which need to be freed
     */
    void free(Storage& storage);
    /**
     * convert bin index to mininal chunk size int the bin, tool func used when alloc
     * @param index : the bin index
     * @return : the minimal size of the chunks in the "index"th bin
     */
    std::size_t index2Size(int index) {
        return static_cast<std::size_t>(256) << index;
    }

    /**
     * Log2FloorNonZero, tool func used when alloc
     * @param n
     * @return Log2FloorNonZero(n)
     */
    inline int Log2FloorNonZero(unsigned n) {
        return 63 ^ __builtin_clzll(n);
    }

    /**
     * find the min binindex for the given size, tool func used when alloc
     * @param bytes : the input size
     * @return : min binindex for the size
     */
    int size2Index(std::size_t bytes) {
        int v = std::max<std::size_t>(bytes, 256);
        int b = std::min(binNum_ - 1, Log2FloorNonZero(v));
        return b;
    }

    /**
     * set the given size to int times of 256, tool func used when alloc
     * @param : bytes input size
     * @return : new size >= input size, and is int times of 256
     */
    std::size_t roundTheSize(std::size_t bytes) {
        std::size_t rounded_bytes =
            (kMinAllocationSize *
             ((bytes + kMinAllocationSize - 1) / kMinAllocationSize));

        return rounded_bytes;
    }

    /**
     * alloc a free chunk from allChunks
     * @return : chunkId (int)
     */
    int allocChunk();
    /**
     * dealloc the chunk, send back to allChunks
     * @param c : chunkId
     */
    void deAllocChunk(int c);
    /**
     * Combine c2 to c1, c2 is freed afterwards
     * @param c1 : the chunk accepting the combine
     * @param c2 : the chunks disappear after combine
     * @return : true if successfully combine, false if c1 c2 not physically connected
     */
    bool combine(int c1, int c2);
    /**
     * insert a free chunk to allChunks, if physically connected with other
     * @param chunkId
     */
    void insertNewChunk(int chunkId); // this chunk is free ( when release a chunk or add new physical memory)
    /**
     * split the big chunk into 2 small chunk, the first is to fit the bytes, the second is to be insert
     * into freeChunkSet
     * @param chunkId : the chunk to be split
     * @param bytes : needed bytes
     */
    void splitChunk(int chunkId, std::size_t bytes); // c should be modified
    /**
     * when mmanager has no free chunk to meet the req, use divice malloc to alloc more memory
     * @param bytes : the size(rounded) of the req
     * @return : the chunk id containing newly malloc memory
     */
    int extend(std::size_t bytes);
    /**
     * find the most suitable memory for the given bytes, may use split or extend
     * @param bytes : the req size
     * @return : the chunk id containing the target chunk
     */
    int findFreeChunk(std::size_t bytes);

    std::size_t requestedMemory() { // for test use
        std::size_t ans = 0;
        for(auto it : allChunks) {
            ans += it.requestedSize;
        }
        return ans;
    }

    void show(); // for test use

    bool everyThingMerged() { // for test use
        std::unordered_set<void*> tempSet;
        int cnt = 0;
        for(int i = 0;i<allChunks.size();++i) {
            if(!allChunks[i].inUse() && allChunks[i].size > 0) {
                tempSet.insert(allChunks[i].ptr);
                tempSet.insert(static_cast<void*>(static_cast<char*>(allChunks[i].ptr) + allChunks[i].size));
                ++cnt;
            }
        }
        if(cnt * 2 == tempSet.size()) return true;
        else return false;
    }

};




#endif //GARLIC_BFC_MMANAGER_H
