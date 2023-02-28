#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "image.h"

CIVL::Image CIVL::OpenImage(const char *filename) {
    int x, y, n;
    stbi_info(filename, &x, &y, &n);
    unsigned char* data = stbi_load(filename, &x, &y, &n, 4);

    Image image;
    image.width = x;
    image.height = y;
    image.pixels.resize(x * y);

    for(int i = 0; i < x * y; i++){
        image.pixels[i].r = data[i * 4 + 0];
        image.pixels[i].g = data[i * 4 + 1];
        image.pixels[i].b = data[i * 4 + 2];
        image.pixels[i].a = data[i * 4 + 3];
    }

    return image;
}