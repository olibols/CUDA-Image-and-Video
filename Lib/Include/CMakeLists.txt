set(INCLUDE_SOURCE_FILES
        library.cu
        testmultiply.cuh testmultiply.cu
        ImageContainer.cuh ImageContainer.cu DataParser.cuh)


add_library(CIVL STATIC
        ${INCLUDE_SOURCE_FILES})
