set(INCLUDE_SOURCE_FILES
        stb_image.h
        stb_image_write.h
        image.h
        image.cu
        image_encoder.h
        image_encoder.cu Processors/blur.h Processors/blur.cu Processors/edge.h Processors/edge.cu Processors/average.h Processors/average.cu Processors/sharpen.h Processors/sharpen.cu)

# add_subdirectory(ImageParsing)

add_library(CIVL STATIC
        ${INCLUDE_SOURCE_FILES})
