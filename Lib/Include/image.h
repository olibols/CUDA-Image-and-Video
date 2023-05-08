#ifndef CIVL_IMAGE_H
#define CIVL_IMAGE_H

#include <vector>

namespace CIVL {

    /// \brief Pixel struct
    /// \details Pixel struct with red, green, blue and alpha channels
    struct Pixel {
        /// \brief Red channel
        unsigned char r = 0;
        /// \brief Green channel
        unsigned char g = 0;
        /// \brief Blue channel
        unsigned char b = 0;
        /// \brief Alpha channel
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

    /// \brief Image struct
    /// \details Image struct with width, height and pixel data
    struct Image {
        /// \brief Width of the image
        int width = 0;
        /// \brief Height of the image
        int height = 0;
        /// \brief Pixel data
        std::vector<Pixel> pixels;

        /// \brief Crop the image
        /// \param x X coordinate of the top left corner of the crop region
        /// \param y Y coordinate of the top left corner of the crop region
        /// \param w Width of the crop region
        /// \param h Height of the crop region
        /// \return Cropped image
        Image crop(int x, int y, int w, int h);

        /// \brief Resize the image
        /// \param w New width of the image
        /// \param h New height of the image
        /// \return Resized image
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

    /// \brief Open an image from a file
    /// \param filename Filename of the image
    /// \return Image
    Image OpenImage(const char *filename);
} // CIVL

#endif //CIVL_IMAGE_H
