#ifndef IMAGE_ENCODER_H
#define IMAGE_ENCODER_H

#include "image.h"

namespace CIVL {
    class ImageEncoder {
    public:
        ImageEncoder();

        ~ImageEncoder();

        /// \brief Encode an image to a file
        /// \param filename Filename to encode to
        /// \param image Image to encode
        static void EncodeImage(const char *filename, Image image);
    };
} // CIVL

#endif