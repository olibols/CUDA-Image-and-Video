#ifndef CIVL_BLUR_H
#define CIVL_BLUR_H

#include "../image.h"

namespace CIVL { namespace Blur{
    Image BoxBlur(Image image, int radius);
} }

#endif //CIVL_BLUR_H
