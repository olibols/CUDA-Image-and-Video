#ifndef CIVL_IMAGE_H
#define CIVL_IMAGE_H

#include <vector>

namespace CIVL {

    struct Pixel {
        unsigned char r = 0;
        unsigned char g = 0;
        unsigned char b = 0;
        unsigned char a = 0;

        Pixel operator+(int other);
        Pixel operator+(Pixel other);
        Pixel operator*(float other);
    };

    struct Image {
        int width = 0;
        int height = 0;
        std::vector<Pixel> pixels;
    };

    Image OpenImage(const char *filename);
} // CIVL

#endif //CIVL_IMAGE_H
