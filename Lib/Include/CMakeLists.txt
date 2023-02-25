set(INCLUDE_SOURCE_FILES
        library.cu
        stb_image.h
        stb_image_write.h
        )

add_subdirectory(ImageParsing)

add_library(CIVL STATIC
        ${INCLUDE_SOURCE_FILES})
