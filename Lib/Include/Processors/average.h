#ifndef CIVL_AVERAGE_H
#define CIVL_AVERAGE_H

#include "../image.h"

namespace CIVL { namespace Average {
    Image WeightedAverage(Image image1, Image image2, float amount);
    Image Average(Image image1, Image image2);
    Image ScreenBlend(Image image1, Image image2);
} }

#endif //CIVL_AVERAGE_H
