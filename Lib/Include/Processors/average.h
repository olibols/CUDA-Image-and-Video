#ifndef CIVL_AVERAGE_H
#define CIVL_AVERAGE_H

#include "../image.h"

namespace CIVL { namespace Average {
    /// \brief Weighted average of two images
    /// \param image1 First image
    /// \param image2 Second image
    /// \param amount Weight of the first image in the average
    /// \return Weighted average of the two images
    Image WeightedAverage(Image image1, Image image2, float amount);

    /// \brief Average of two images
    /// \param image1 First image
    /// \param image2 Second image
    /// \return Average of the two images
    Image Average(Image image1, Image image2);

    /// \brief Screen blend mode of two images
    /// \param image1 First image
    /// \param image2 Second image
    /// \return Blended image
    Image ScreenBlend(Image image1, Image image2);
} }

#endif //CIVL_AVERAGE_H
