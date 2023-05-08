#include "sharpen.h"
#include "blur.h"

namespace CIVL{
    namespace Sharpen{
        Image UnsharpMask(Image image, float amount){
            // Box blur the image
            Image blurred = Blur::BoxBlur(image, 2);
            // Subtract the blurred image from the original
            Image diff = image - blurred;
            // Add the difference to the original
            return image + (diff * amount);
        }
    }
}