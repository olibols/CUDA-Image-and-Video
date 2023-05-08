//
// Created by olibo on 07/11/2022.
//

#include <stdio.h>
#include <random>
#include <thread>
#include "image.h"
#include "image_encoder.h"
#include "Processors/blur.h"
#include "Processors/edge.h"
#include "Processors/average.h"
#include "Processors/sharpen.h"

int main(){
    CIVL::Image image1 = CIVL::OpenImage("C:\\Users\\olibo\\Downloads\\avg1.png");
    CIVL::Image image2 = CIVL::OpenImage("C:\\Users\\olibo\\Downloads\\avg2.png");

    //CIVL::Image averageImage = CIVL::Average::ScreenBlend(image1, image2);
    //CIVL::Image resized = image1.resize(1000, 1000);

    CIVL::Image testsharpened = CIVL::OpenImage("C:\\Users\\olibo\\Downloads\\test2.png");
    CIVL::Image sharpened = CIVL::Sharpen::UnsharpMask(testsharpened, 10);
    //CIVL::Image resized = testsharpened.resize(500, 500);
    CIVL::ImageEncoder::EncodeImage("C:\\Users\\olibo\\Downloads\\testimage.jpg", sharpened);
}