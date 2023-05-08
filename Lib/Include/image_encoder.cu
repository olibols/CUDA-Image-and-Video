#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "image_encoder.h"

CIVL::ImageEncoder::ImageEncoder() = default;

CIVL::ImageEncoder::~ImageEncoder() = default;

void CIVL::ImageEncoder::EncodeImage(const char *filename, Image image) {
    // Convert image.pixels to unsigned char*
    unsigned char* data = new unsigned char[image.width * image.height * 4];
    for(int i = 0; i < image.width * image.height; i++){
        data[i * 4 + 0] = image.pixels[i].r;
        data[i * 4 + 1] = image.pixels[i].g;
        data[i * 4 + 2] = image.pixels[i].b;
        data[i * 4 + 3] = image.pixels[i].a;
    }

    // Save image
    stbi_write_png(filename, image.width, image.height, 4, data, image.width * 4);
}