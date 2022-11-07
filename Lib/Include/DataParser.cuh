//
// Created by olibo on 07/11/2022.
//

#ifndef CIVL_DATAPARSER_CUH
#define CIVL_DATAPARSER_CUH

template<class T>
class DataParser {
public:
    DataParser();
    ~DataParser();

    virtual T parse(const char* pPath);
    virtual void save(const char* pPath);
    virtual void save(const char* pPath, T* pData);
};


#endif //CIVL_DATAPARSER_CUH
