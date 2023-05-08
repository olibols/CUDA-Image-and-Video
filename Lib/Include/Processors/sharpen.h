#ifndef CIVL_SHARPEN_H
#define CIVL_SHARPEN_H

#include "../image.h"

namespace CIVL { namespace Sharpen{
    /// \brief Unsharp mask sharpening an image
    /// \param image Image to sharpen
    /// \param amount Amount of sharpening
    /// \return Sharpened image
    Image UnsharpMask(Image image, float amount);
} }

#endif //CIVL_SHARPEN_H
