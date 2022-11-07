//
// Created by olibo on 07/11/2022.
//
#include "ImageContainer.cuh"

namespace CIVL{
    ImageContainer::ImageContainer() {
        m_pixels = nullptr;
        m_width = 0;
        m_height = 0;
    }

    ImageContainer::~ImageContainer() {
        delete[] m_pixels;
    }

    void ImageContainer::load(const char* pPath) { // Load image from file

    }

    void ImageContainer::save(const char* pPath) {

    }

    Pixel ImageContainer::getPixel(int x, int y) {
        return m_pixels[y * m_width + x];
    }

    Pixel* ImageContainer::getPixels() {
        return m_pixels;
    }

    int ImageContainer::getWidth() const {
        return m_width;
    }

    int ImageContainer::getHeight() const {
        return m_height;
    }

    void ImageContainer::setPixel(int x, int y, Pixel pixel) {
        m_pixels[y * m_width + x] = pixel;
    }

    void ImageContainer::setPixels(Pixel *pPixels) {
        this->m_pixels = pPixels;
    }
}
