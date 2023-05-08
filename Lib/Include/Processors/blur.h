#ifndef CIVL_BLUR_H
#define CIVL_BLUR_H

#include "../image.h"

namespace CIVL { namespace Blur{

    /// \brief Box blur an image
    /// \param image Image to blur
    /// \param radius Radius of the blur
    /// \return Blurred image
    Image BoxBlur(Image image, int radius);
} }

#endif //CIVL_BLUR_H
