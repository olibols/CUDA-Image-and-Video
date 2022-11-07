//
// Created by olibo on 07/11/2022.
//

#ifndef CIVL_IMAGEPARSER_CUH
#define CIVL_IMAGEPARSER_CUH

#include "DataParser.cuh"

namespace CIVL {
    class Pixel;

    class ImageParser : public DataParser {
    public:
        ImageParser(const char *pPath); // Constructor
        ~ImageParser(); // Destructor

        Pixel* latestImage() override; // Get latest image
    };
}

#endif //CIVL_IMAGEPARSER_CUH
