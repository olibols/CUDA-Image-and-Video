#ifndef CIVL_EDGE_H
#define CIVL_EDGE_H

#include "../image.h"

namespace CIVL { namespace Edge{

    /// \brief Sobel edge detection
    /// \param image Image to detect edges in
    /// \param threshold Threshold for edge detection
    /// \return Image with edges
    Image Sobel(Image image, int threshold);
} }

#endif //CIVL_EDGE_H
