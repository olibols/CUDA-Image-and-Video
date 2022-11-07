set(INCLUDE_SOURCE_FILES
        library.cu
        testmultiply.cuh testmultiply.cu
        ImageParsing/ImageContainer.cuh ImageParsing/ImageContainer.cu
        ImageParsing/DataParser.cuh ImageParsing/ImageParser.cu ImageParsing/ImageParser.cuh)

add_subdirectory(ImageParsing)

add_library(CIVL STATIC
        ${INCLUDE_SOURCE_FILES})
