//
// Created by olibo on 07/11/2022.
//

#ifndef CIVL_DATAPARSER_CUH
#define CIVL_DATAPARSER_CUH

namespace CIVL {
    enum StreamType {
        IMAGE,
        VIDEO,
    };

    struct Pixel;

    class DataParser {
    public:
        inline DataParser(StreamType type, const char *pPath) {
            this->m_type = type;
            this->pPath = pPath;
        }; // Constructor

        ~DataParser();

        virtual Pixel* latestImage(); // Get latest image

        inline const char* getPath() { return this->pPath;} // Get path to data
        inline StreamType getStreamType() { return m_type; }; // Get stream type

    protected:
        StreamType m_type; // Type of stream
        const char *pPath; // Path to file

    };
}
#endif //CIVL_DATAPARSER_CUH
