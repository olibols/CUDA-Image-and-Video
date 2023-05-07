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

int main(){
    CIVL::Image image1 = CIVL::OpenImage("C:\\Users\\olibo\\Downloads\\avg1.png");
    CIVL::Image image2 = CIVL::OpenImage("C:\\Users\\olibo\\Downloads\\avg2.png");

    CIVL::Image averageImage = CIVL::Average::ScreenBlend(image1, image2);
    //CIVL::Image averageImage = CIVL::Edge::Sobel(image1, 10);

    CIVL::ImageEncoder::EncodeImage("C:\\Users\\olibo\\Downloads\\test2.png", averageImage);
}