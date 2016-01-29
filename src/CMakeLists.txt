set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O2  --ptxas-options=-v -g -G  -gencode=arch=compute_30,code=sm_30
        -gencode=arch=compute_35,code=sm_35
        -gencode=arch=compute_50,code=sm_50
        -gencode=arch=compute_50,code=compute_50)
#set(CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)

# Find external libraries and dependencies
#set( OpenCV_DIR /home/prenav/sbox/image-core/opencv-3.0.0/build)
set( OpenCV_DIR /home/prenav/sbox/image-core/opencv-3.0.0/build)
find_package(OpenCV REQUIRED NO_DEFAULT_PATH )
find_package(CUDA REQUIRED)

if (NOT APPLE)
  find_package(OpenMP)
endif ()

# ============================================================================ #
# Compilation flags
IF(UNIX)
  SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} -g -O2  -Wall -Wextra -Wunused-variable -DDEBUG -D_DEBUG")
  SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O2 -g  -Wall -Wextra -Wunused-variable -DDEBUG -D_DEBUG")
ENDIF(UNIX)

if(OPENMP_FOUND)
  MESSAGE("OpenMP found")
  if(UNIX)
    SET(CMAKE_C_FLAGS_RELEASE "-O2  -Wall -Wextra -Wunused-variable  -g -fPIC -msse2 -msse3 -msse4 -ffast-math")
    SET(CMAKE_CXX_FLAGS_RELEASE "-O2 -Wall -Wextra -Wunused-variable -g -fPIC -msse2 -msse3 -msse4 -ffast-math")
  endif(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
else(OPENMP_FOUND)
    MESSAGE("OpenMP not found")
    if(UNIX)
      SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS} -O2 -Wall -std=c++0x -Wunused-variable -Wno-unknown-pragmas -g -fPIC -msse2 -msse3 -msse4 -ffast-math")
      SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O2 -Wall -std=c++0x -Wno-unknown-pragmas -Wunused-variable -g -fPIC -msse2 -msse3 -msse4 -ffast-math")
    endif(UNIX)
endif(OPENMP_FOUND)

# ============================================================================ #
# Create projects
include_directories("lib/" ${OpenCV_INCLUDE_DIRS})

# AKAZE libraries
set(AKAZE_SRCS
    lib/AKAZEConfig.h
    lib/AKAZE.h                  lib/AKAZE.cpp
    lib/fed.h                    lib/fed.cpp
    lib/nldiffusion_functions.h  lib/nldiffusion_functions.cpp
    lib/utils.h                  lib/utils.cpp)

set(CUDA_SEPARABLE_COMPILATION ON)
cuda_add_library(AkazeCuda gpu/ImageOperation.h gpu/GeneralFilters.cu gpu/FlowFilter.cu gpu/GaussianFilter.cu gpu/Descriptor.cu)

add_library(AKAZE ${AKAZE_SRCS})
target_link_libraries(AKAZE ${OpenCV_LIBS} AkazeCuda)

# Feature detector program
add_executable(akaze_features akaze_features.cpp)
target_link_libraries(akaze_features AKAZE )

# Image matching program
add_executable(akaze_match akaze_match.cpp)
target_link_libraries(akaze_match AKAZE )

# Program that compares AKAZE with ORB and BRISK
add_executable(akaze_compare akaze_compare.cpp)
target_link_libraries(akaze_compare AKAZE )

# ============================================================================ #
# Library installation
install(TARGETS AKAZE DESTINATION ${AKAZE_INSTALL_PREFIX})
install(TARGETS AkazeCuda DESTINATION ${AKAZE_INSTALL_PREFIX})
install(FILES
    lib/AKAZE.h
    lib/fed.h
    lib/utils.h
    lib/nldiffusion_functions.h
    lib/AKAZEConfig.h
    gpu/ImageOperation.h
    gpu/helper_cuda.h
    gpu/helper_cuda_drvapi.h
    gpu/helper_cuda_gl.h
    gpu/helper_device_thread.h
    gpu/helper_functions.h
    gpu/helper_math.h
    gpu/helper_image.h
    gpu/helper_timer.h
    gpu/helper_string.h
    gpu/exception.h
    DESTINATION
    ${AKAZE_INCLUDE_PREFIX}
)

# ============================================================================ #
