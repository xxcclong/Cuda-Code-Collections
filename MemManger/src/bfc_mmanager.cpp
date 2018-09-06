#include <bfc_mmanager.h>

BFCMemManager::BFCMemManager(std::size_t allMem) :
    allMemory_(allMem) {
    availableMemory = 500 << 20;
//    std::cout<< availableMemory << std::endl;
    if (allMemory_ == 0 || availableMemory == 0) {
        // log error
        printf("allMemory must be set\n");
        return;
    }
    if (availableMemory == allMem) {
        // log info
        printf("no growth mode, total memory %zu\n", availableMemory);
    }

    // set binNum according to availableMemory
    std::size_t tempMem = availableMemory >> 8;
    binNum_ = 0;
    while (tempMem > 0) {
        tempMem >>= 1;
        binNum_ += 1;
    }

    // init bins
    bins.resize(binNum_);
    // init chunks and free chunk Ids
    allChunks.resize(maxChunkNum_);
    for (int i = 0; i < maxChunkNum_; ++i)
        freeChunks.push(i);
    // init original memory,
    int initC = allocChunk();
    Chunk* initChunk = &(allChunks[initC]);
#ifdef CUDA
    cudaError_t err = (cudaMalloc(&(initChunk->ptr), allMemory_));
    if(cudaSuccess != err) fprintf(stderr, "fail to alloc %zu  %s \n",allMemory_, cudaGetErrorString(err));
#else
    initChunk->ptr = malloc(allMemory_);
#endif
    allocatedMemoryPtrs.push_back(initChunk->ptr);
    allocatedMemoryBytes.push_back(allMemory_);
    initChunk->size = allMemory_;
    insertNewChunk(initC);
}

BFCMemManager::~BFCMemManager() {
#ifdef DEBUG
    printf("begin ~bfc_mmanager\n");
#endif
    for (int i = 0; i < allocatedMemoryPtrs.size(); ++i) {
        // free 
#ifdef DEBUG
        printf("freeing %d chunk with ptr %p\n", i, allocatedMemoryPtrs[i]);
#endif
#ifdef CUDA
        cudaFree(allocatedMemoryPtrs[i]);
#else 
        std::free(allocatedMemoryPtrs[i]);
#endif
    }
}

Storage* BFCMemManager::alloc(std::size_t nBytes, unsigned int hint) {
    int whichC = findFreeChunk(nBytes);
    return new Storage(allChunks[whichC].ptr, nBytes, whichC);
}

void BFCMemManager::free(Storage& storage) {
    if(allChunks.size() <=storage.whichC)
    {
        printf("whichC is %d while allChunks has size %d\n", storage.whichC, allChunks.size());
        return;
    }
    if(!allChunks[storage.whichC].inUse())
    {
        printf("%d CHunk is not in use, cannot free\n", storage.whichC);
        return;
    }
    insertNewChunk(storage.whichC);
}

int BFCMemManager::findFreeChunk(std::size_t bytes) {
    std::size_t bytesInNeed = roundTheSize(bytes);
    int whichBin = size2Index(bytesInNeed);
    for (; whichBin < binNum_; ++whichBin) {
        for (auto cIter = bins[whichBin].freeChunks.begin(); cIter != bins[whichBin].freeChunks.end();
                ++cIter) {
            Chunk* c = *cIter;
            int whichC = c->chunkId;
            if (!c->inUse() && c->size >= bytesInNeed) { // get the chunk, remove from the bin free set
                bins[whichBin].removeIter(cIter);
                c->allocId = allocTimes_;
                allocTimes_++;
                const int kMaxInternalFragmentation = 128 << 20;  // 128mb
                if (c->size >= bytesInNeed * 2 ||
                        static_cast<int>(c->size) - bytesInNeed >=
                        kMaxInternalFragmentation) { // < half of the chunk size or have gap size > 128mb
                    splitChunk(whichC, bytesInNeed);
                }
                c->requestedSize = bytes;
                return whichC;
            }
        }
    }
    int whichC = extend(bytesInNeed);
    Chunk* c = &(allChunks[whichC]);
    c->allocId = allocTimes_;
    allocTimes_++;
    const int kMaxInternalFragmentation = 128 << 20;  // 128mb
    if (c->size >= bytesInNeed * 2 ||
            static_cast<int>(c->size) - bytesInNeed >=
            kMaxInternalFragmentation) {
        splitChunk(whichC, bytesInNeed);
    }
    c->requestedSize = bytes;
    c->allocId = allocTimes_;
    allocTimes_ += 1;
    return whichC;
}

int BFCMemManager::extend(std::size_t bytes) {
    if (bytes > availableMemory - allMemory_) {
        // log
        printf("no enough memory\n");
        return -1;
    }
    std::size_t addedMemory = 0;
    if (allMemory_ * 2 < availableMemory) {
        addedMemory = std::max(allMemory_, bytes);
    } else {
        std::size_t beishu = allMemory_;
        while (beishu + allMemory_ > availableMemory) {
            beishu *= 0.9;
        }
        addedMemory = std::max(beishu, bytes);
    }
    int ans = allocChunk();
    Chunk* ansChunk = &(allChunks[ans]);
    ansChunk->physicalNext = -1;
    ansChunk->physicalPrev = -1;
#ifdef CUDA
    cudaError_t err = (cudaMalloc(&(ansChunk->ptr), addedMemory));
    if(cudaSuccess != err) fprintf(stderr, "fail to alloc %zu  %s \n",addedMemory,  cudaGetErrorString(err));
#else
    ansChunk->ptr = malloc(addedMemory);
#endif
    ansChunk->size = addedMemory;
    allMemory_ += addedMemory;
    allocatedMemoryPtrs.push_back(ansChunk->ptr);
    allocatedMemoryBytes.push_back(addedMemory);
    if (ansChunk->ptr == nullptr) {
        // log
        printf("fail to allocate %zu memory\n", addedMemory);
        return -1;
    }
    ansChunk->allocId = -1;
    return ans;
}

void BFCMemManager::splitChunk(int whichChunk, std::size_t bytes) {
    Chunk* thisC = &(allChunks[whichChunk]);
    int newChunk = allocChunk();
    Chunk* newC = &(allChunks[newChunk]);

    newC->physicalPrev = whichChunk;
    newC->physicalNext = thisC->physicalNext;
    thisC->physicalNext = newChunk;
    newC->size = thisC->size - bytes;
    newC->requestedSize = 0;
    newC->allocId = -1;
    newC->ptr = static_cast<void*>(static_cast<char*>(thisC->ptr) + bytes);

    thisC->size = bytes;
    thisC->requestedSize = bytes;
    insertNewChunk(newChunk);
}

void BFCMemManager::insertNewChunk(int whichChunk) {
    Chunk* c = &(allChunks[whichChunk]);
    if (c->physicalPrev != -1 && !allChunks[c->physicalPrev].inUse()) {
        bins[allChunks[c->physicalPrev].whichBin].removeChunkId(&(allChunks[c->physicalPrev]));
        int toBeDeAlloc = c->physicalPrev;
        if (combine(whichChunk, c->physicalPrev))
            deAllocChunk(toBeDeAlloc);
    }
    if (c->physicalNext != -1 && !allChunks[c->physicalNext].inUse()) {
        bins[allChunks[c->physicalNext].whichBin].removeChunkId(&(allChunks[c->physicalNext]));
        int toBeDeAlloc = c->physicalNext;
        if (combine(whichChunk, c->physicalNext))
            deAllocChunk(toBeDeAlloc);
    }
    // now c is ready
    int whichBin = size2Index(c->size);
    bins[whichBin].freeChunks.insert(c);
    c->whichBin = whichBin;
    c->requestedSize = 0;
    c->allocId = -1;
}

bool BFCMemManager::combine(int c1, int c2) {
    Chunk* chunk1 = &(allChunks[c1]);
    Chunk* chunk2 = &(allChunks[c2]);
    if (chunk1->physicalPrev == c2) {
        chunk1->physicalPrev = chunk2->physicalPrev;
        chunk1->ptr = chunk2->ptr;
    } else if (chunk1->physicalNext == c2) {
        chunk1->physicalNext = chunk2->physicalNext;
    } else {
        //log
        printf("chunk %d and %d are not physically connected\n", c1, c2);
        return false;
    }
    chunk1->size += chunk2->size;
    chunk1->requestedSize += chunk2->requestedSize;
    return true;

}

int BFCMemManager::allocChunk() {
    if (freeChunks.empty()) {
        allChunks.resize(maxChunkNum_ * 2);
        for (int i = 0; i < maxChunkNum_; ++i) {
            freeChunks.push(i + maxChunkNum_);
        }
        maxChunkNum_ *= 2;
    }
    int ans = freeChunks.front();
    freeChunks.pop();
    allChunks[ans].chunkId = ans;
    return ans;
}

void BFCMemManager::deAllocChunk(int whichChunk) {
    allChunks[whichChunk].chunkId = -1;
    allChunks[whichChunk].allocId = -1;
    allChunks[whichChunk].size = 0;
    allChunks[whichChunk].requestedSize = 0;
    allChunks[whichChunk].whichBin = -1;
    freeChunks.push(whichChunk);
}
void BFCMemManager::show() {
    std::cout << "all malloc memorys "<< allMemory_ << std::endl;
    for(int i = 0;i<allocatedMemoryPtrs.size();++i)
    {
        std::cout << allocatedMemoryPtrs[i]<< std::endl;
        std::cout << allocatedMemoryBytes[i]<< std::endl;
    }
    for (int i = 0; i < maxChunkNum_; ++i) {
        if (allChunks[i].inUse()) {
            std::cout << "in use chunk:" << std::endl;
            std::cout << "chunk " << i << " size " << allChunks[i].size << " prev " << allChunks[i].physicalPrev \
                      << " next " << allChunks[i].physicalNext << " place " << allChunks[i].ptr << " requested size " << \
                      allChunks[i].requestedSize << std::endl;
        } else if (allChunks[i].size != 0) {
            std::cout << "free chunk:" << std::endl;
            std::cout << "chunk " << i << " size " << allChunks[i].size << " prev " << allChunks[i].physicalPrev \
                      << " next " << allChunks[i].physicalNext << " place " << allChunks[i].ptr << " requested size " << \
                      allChunks[i].requestedSize << std::endl;
        }
    }
    std::cout << "All free Chunks binnum " << binNum_ << std::endl;
    for (int i = 0; i < binNum_; ++i)
        for (auto cIter = bins[i].freeChunks.begin(); cIter != bins[i].freeChunks.end();
                ++cIter) {
            Chunk* c = *cIter;
            std::cout << c->size << " in which bin "<< i << std::endl;
        }
    return;
}



