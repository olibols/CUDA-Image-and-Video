#ifndef CIVL_SHARPEN_H
#define CIVL_SHARPEN_H

#include "../image.h"

namespace CIVL { namespace Sharpen{
        Image UnsharpMask(Image image, float amount);
} }

#endif //CIVL_SHARPEN_H
