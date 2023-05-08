#include "sharpen.h"
#include "blur.h"

namespace CIVL{
    namespace Sharpen{
        Image UnsharpMask(Image image, float amount){
            Image blurred = Blur::BoxBlur(image, 2);
            Image diff = image - blurred;
            return image + (diff * amount);
        }
    }
}