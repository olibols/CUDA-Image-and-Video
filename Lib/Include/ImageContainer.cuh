//
// Created by olibo on 07/11/2022.
//

#ifndef CIVL_IMAGECONTAINER_CUH
#define CIVL_IMAGECONTAINER_CUH

namespace CIVL{

    struct Pixel{
        unsigned char r;
        unsigned char g;
        unsigned char b;
        unsigned char a;
    };

    class ImageContainer{
    public:
        // Constructor and destructor methods
        ImageContainer();
        ~ImageContainer();

        void load(const char* pPath); // Load image from file
        void save(const char* pPath); // Save image to file

        // Getters and setters
        Pixel getPixel(int x, int y); // Get pixel at position (x, y)
        Pixel* getPixels(); // Get all pixels
        int getWidth() const; // Get image width
        int getHeight() const; // Get image height
        void setPixel(int x, int y, Pixel pixel); // Set pixel at position (x, y)
        void setPixels(Pixel* pPixels); // Set all pixels

    private:
        Pixel* m_pixels; // Array of pixels
        int m_width; // Width of image
        int m_height; // Height of image
    };
}

#endif //CIVL_IMAGECONTAINER_CUH
