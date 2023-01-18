#ifndef CIVL_EDGE_H
#define CIVL_EDGE_H

#include "../image.h"

namespace CIVL { namespace Edge{
    Image Sobel(Image image, int threshold);
} }

#endif //CIVL_EDGE_H
