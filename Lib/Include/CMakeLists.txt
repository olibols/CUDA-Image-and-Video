set(INCLUDE_SOURCE_FILES
        library.cu
        testmultiply.cuh
        testmultiply.cu)


add_library(CIVL STATIC
        ${INCLUDE_SOURCE_FILES})
