#ifndef IMAGE_ENCODER_H
#define IMAGE_ENCODER_H

#include "image.h"

namespace CIVL {
    class ImageEncoder {
    public:
        ImageEncoder();

        ~ImageEncoder();

        static void EncodeImage(const char *filename, Image image);
    };
} // CIVL

#endif