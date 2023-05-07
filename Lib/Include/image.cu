#define STB_IMAGE_IMPLEMENTATION

#include <stdexcept>
#include "stb_image.h"
#include <algorithm>
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

CIVL::Image CIVL::Image::crop(int x, int y, int w, int h) {
    // Check if the crop dimensions are valid
    if (x < 0 || y < 0 || x + w > width || y + h > height) {
        throw std::runtime_error("Invalid crop dimensions.");
    }

    Image cropped;
    cropped.width = w;
    cropped.height = h;
    cropped.pixels.resize(w * h);

    // Iterate over the pixels in the crop region
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            // Copy the pixel from the original image to the cropped image
            cropped.pixels[i * w + j] = pixels[(y + i) * width + (x + j)];
        }
    }
    return cropped;
}

CIVL::Pixel BilinearInterpolate(float x, float y, CIVL::Image image){
    int x1 = (int) x;
    int y1 = (int) y;
    int x2 = std::min(x1 + 1, image.width - 1);
    int y2 = std::min(y1 + 1, image.height - 1);

    float dx = x - x1;
    float dy = y - y1;

    CIVL::Pixel p1 = image.pixels[y1 * image.width + x1];
    CIVL::Pixel p2 = image.pixels[y1 * image.width + x2];
    CIVL::Pixel p3 = image.pixels[y2 * image.width + x1];
    CIVL::Pixel p4 = image.pixels[y2 * image.width + x2];

    CIVL::Pixel top = p1 * (1 - dx) + p2 * dx;
    CIVL::Pixel bottom = p3 * (1 - dx) + p4 * dx;

    return top * (1 - dy) + bottom * dy;
}

CIVL::Image CIVL::Image::resize(int w, int h) {
    Image result;
    result.width = w;
    result.height = h;
    result.pixels.resize(w * h);

    float scaleX = (float) width / w;
    float scaleY = (float) height / h;

    for(int y = 0; y < h; ++y){
        for(int x = 0; x < w; ++x){
            float srcX = x * scaleX;
            float srcY = y * scaleY;

            result.pixels[y * w + x] = BilinearInterpolate(srcX, srcY, *this);
        }
    }

    return result;
}
