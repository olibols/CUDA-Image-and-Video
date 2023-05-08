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
                    255};
        }

        Pixel operator+(Pixel other) {
            return {static_cast<unsigned char>(r + other.r),
                    static_cast<unsigned char>(g + other.g),
                    static_cast<unsigned char>(b + other.b),
                    255};
        }

        Pixel operator*(float other) {
            return {static_cast<unsigned char>(r * other),
                    static_cast<unsigned char>(g * other),
                    static_cast<unsigned char>(b * other),
                    255};
        }

        Pixel operator-(Pixel other){
            return {static_cast<unsigned char>(r - other.r),
                    static_cast<unsigned char>(g - other.g),
                    static_cast<unsigned char>(b - other.b),
                    255};
        }
    };

    struct Image {
        int width = 0;
        int height = 0;
        std::vector<Pixel> pixels;

        Image crop(int x, int y, int w, int h);

        Image resize(int w, int h);

        Image operator-(Image other) {
            Image result;
            result.width = width;
            result.height = height;
            result.pixels.resize(width * height);

            for (int i = 0; i < width * height; i++) {
                result.pixels[i] = pixels[i] - other.pixels[i];
            }

            return result;
        };

        Image operator*(float other){
            Image result;
            result.width = width;
            result.height = height;
            result.pixels.resize(width * height);

            for (int i = 0; i < width * height; i++) {
                result.pixels[i] = pixels[i] * other;
            }

            return result;
        }

        Image operator+(Image other){
            Image result;
            result.width = width;
            result.height = height;
            result.pixels.resize(width * height);

            for (int i = 0; i < width * height; i++) {
                result.pixels[i] = pixels[i] + other.pixels[i];
            }

            return result;
        }
    };


    Image OpenImage(const char *filename);
} // CIVL

#endif //CIVL_IMAGE_H
