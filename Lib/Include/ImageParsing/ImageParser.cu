//
// Created by olibo on 07/11/2022.
//

#include "ImageParser.cuh"

namespace CIVL{
    /***
     * Constructor
     * @param pPath Path to the image file
     */
    ImageParser::ImageParser(const char *pPath) : DataParser(StreamType::IMAGE, pPath) {
        //m_parser = new DataParser(pPath);
    }

    /***
     * Destructor
     */
    ImageParser::~ImageParser() {
        //delete m_parser;
    }

    /***
     * Get latest image
     * @return Pixel* most recent image
     */
    Pixel* ImageParser::latestImage() {
        return nullptr;
    }
}

