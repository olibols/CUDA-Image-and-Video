set(INCLUDE_SOURCE_FILES
        library.cu
        )

add_subdirectory(ImageParsing)

add_library(CIVL STATIC
        ${INCLUDE_SOURCE_FILES})
