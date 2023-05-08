
# CUDA Image Library

Welcome to the CUDA Image Library! This library provides a suite of image processing functions optimized for high performance using NVIDIA's CUDA technology. In this README, we'll walk you through how to use this library with some simple examples.

## Table of Contents

-   [Installation](#installation)
-   [Usage](#usage)
    -   [Loading and Saving Images](#loading-and-saving-images)
    -   [Image Operations](#image-operations)
        -   [Average](#average)
        -   [Blur](#blur)
        -   [Edge Detection](#edge-detection)
        -   [Sharpen](#sharpen)
    -   [Miscellaneous Image Utilities](#miscellaneous-image-utilities)
-   [Example Code](#example-code)

## Installation

To install the CUDA Image Library, simply clone the repository to your local machine and make sure you have the required dependencies (CUDA, CMake, etc.) installed.

    git clone https://github.com/your_username/cuda-image-library.git
    cd cuda-image-library

## Usage
### Loading and Saving Images

To use the CUDA Image Library, you'll first need to include the necessary header files in your C++ project. Below is a list of handy includes:

    #include "image.h"
    #include "image_encoder.h"
    #include "Processors/blur.h"
    #include "Processors/edge.h"
    #include "Processors/average.h"
    #include "Processors/sharpen.h"

Loading and saving images is easy with the `OpenImage` and `EncodeImage` functions.

    CIVL::Image image = CIVL::OpenImage("path/to/image.png");
    CIVL::ImageEncoder::EncodeImage("path/to/output_image.jpg", image);

### Image Operations

#### Average

The library provides several functions for averaging images:

-   `WeightedAverage`: Combine two images with specified weights.
-   `Average`: Compute the average of two images.
-   `ScreenBlend`: Perform a screen blend operation on two images.

For example

    
# CUDA Image Library

Welcome to the CUDA Image Library! This library provides a suite of image processing functions optimized for high performance using NVIDIA's CUDA technology. In this README, we'll walk you through how to use this library with some simple examples.

## Table of Contents

-   [Installation](#installation)
-   [Usage](#usage)
    -   [Loading and Saving Images](#loading-and-saving-images)
    -   [Image Operations](#image-operations)
        -   [Average](#average)
        -   [Blur](#blur)
        -   [Edge Detection](#edge-detection)
        -   [Sharpen](#sharpen)
    -   [Miscellaneous Image Utilities](#miscellaneous-image-utilities)
-   [Example Code](#example-code)

## Installation

To install the CUDA Image Library, simply clone the repository to your local machine and make sure you have the required dependencies (CUDA, CMake, etc.) installed.

    git clone https://github.com/your_username/cuda-image-library.git
    cd cuda-image-library

## Usage
### Loading and Saving Images

To use the CUDA Image Library, you'll first need to include the necessary header files in your C++ project. Below is a list of handy includes:

    #include "image.h"
    #include "image_encoder.h"
    #include "Processors/blur.h"
    #include "Processors/edge.h"
    #include "Processors/average.h"
    #include "Processors/sharpen.h"

Loading and saving images is easy with the `OpenImage` and `EncodeImage` functions.

    CIVL::Image image = CIVL::OpenImage("path/to/image.png");
    CIVL::ImageEncoder::EncodeImage("path/to/output_image.jpg", image);

### Image Operations

#### Average

The library provides several functions for averaging images:

-   `WeightedAverage`: Combine two images with specified weights.
-   `Average`: Compute the average of two images.
-   `ScreenBlend`: Perform a screen blend operation on two images.

        CIVL::Image averageImage = CIVL::Average::ScreenBlend(image1, image2);

#### Blur

The library currently supports one blur method:

-   `BoxBlur`: Apply a box blur with the specified radius.

	    CIVL::Image blurred = CIVL::Blur::BoxBlur(image, radius);

#### Edge Detection

The library provides one edge detection method:

-   `Sobel`: Apply the Sobel operator for edge detection.

	    CIVL::Image edges = CIVL::Edge::Sobel(image);

#### Sharpen

The library offers one sharpening method:

-   `UnsharpMask`: Apply unsharp masking with the specified radius.

		CIVL::Image sharpened = CIVL::Sharpen::UnsharpMask(image, radius);

### Miscellaneous Image Utilities

The library also includes some general utility functions:

-   `Image.crop(x, y, w, h)`: Crop an image to the specified dimensions.
-   `Image.resize(x, y)`: Resize an image to the specified dimensions.

		CIVL::Image cropped = image.crop(x, y, w, h);
		CIVL::Image resized = image.resize(newWidth, newHeight);

## Example Code

Here's a complete example demonstrating how to use the library:

    #include <stdio.h>
	#include "image.h"
	#include "image_encoder.h"
	#include "Processors/blur.h"
	#include "Processors/edge.h"
	#include "Processors/average.h"
	#include "Processors/sharpen.h"

	int main() {
	    // Load images from files CIVL::Image image1 = CIVL::OpenImage("path/to/image1.png");
	    CIVL::Image image2 = CIVL::OpenImage("path/to/image2.png");
	    
	    // Average two images using screen blend
	    CIVL::Image averageImage = CIVL::Average::ScreenBlend(image1, image2); 
	    
	    // Load a new image for sharpening CIVL::Image testSharpened = 
	    CIVL::OpenImage("path/to/test_image.png"); // Apply unsharp mask to sharpen the image
	    CIVL::Image sharpened = CIVL::Sharpen::UnsharpMask(testSharpened, 10);
	    
	    // Save the processed images to files
	    CIVL::ImageEncoder::EncodeImage("path/to/output_average.jpg", averageImage); 
	    CIVL::ImageEncoder::EncodeImage("path/to/output_sharpened.jpg", sharpened);
	    
	    // Other possible operations (commented out):
	    // CIVL::Image resized = image1.resize(1000, 1000);
	    // CIVL::Image cropped = image.crop(x, y, w, h); 
	}
