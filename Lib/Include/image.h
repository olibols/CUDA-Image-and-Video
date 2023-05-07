#ifndef CIVL_IMAGE_H
#define CIVL_IMAGE_H

#include <vector>

namespace CIVL {

    struct Pixel {
        unsigned char r = 0;
        unsigned char g = 0;
        unsigned char b = 0;
        unsigned char a = 0;

        Pixel operator+(int other) {
            return {static_cast<unsigned char>(r + other),
                    static_cast<unsigned char>(g + other),
                    static_cast<unsigned char>(b + other),
                    static_cast<unsigned char>(a + other)};
        }

        Pixel operator+(Pixel other) {
            return {static_cast<unsigned char>(r + other.r),
                    static_cast<unsigned char>(g + other.g),
                    static_cast<unsigned char>(b + other.b),
                    static_cast<unsigned char>(a + other.a)};
        }

        Pixel operator*(float other) {
            return {static_cast<unsigned char>(r * other),
                    static_cast<unsigned char>(g * other),
                    static_cast<unsigned char>(b * other),
                    static_cast<unsigned char>(a * other)};
        }
    };

    struct Image {
        int width = 0;
        int height = 0;
        std::vector<Pixel> pixels;

        Image crop(int x, int y, int w, int h);
        Image resize(int w, int h);
    };

    Image OpenImage(const char *filename);
} // CIVL

#endif //CIVL_IMAGE_H
