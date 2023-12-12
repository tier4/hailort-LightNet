cmake_minimum_required(VERSION 3.0.0)

find_package(Threads REQUIRED)
set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(HailoRT 4.15.0 EXACT REQUIRED)
find_package(OpenCV REQUIRED)
find_package (gflags REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Ofast -Wall -fopenmp")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -Wall -fopenmp")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O0 -g -Wno-write-strings -fopenmp")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O0 -g -Wall -fopenmp")
#set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb -g")

include_directories (${gflags_INCLUDE_DIR} ./include)

add_executable(hailort-LightNet src/hailort_lightnet.cpp src/common.cpp src/lightnet.cpp src/utils.cpp src/config_parser.cpp src/colormap.cpp)
target_include_directories(hailort-LightNet PRIVATE  ${OpenCV_INCLUDE_DIRS})
target_link_libraries(hailort-LightNet PRIVATE HailoRT::libhailort Threads::Threads ${OpenCV_LIBS} gflags "stdc++fs" "stdc++")

if(WIN32)
    target_compile_options(hailort-LightNet PRIVATE
        /DWIN32_LEAN_AND_MEAN
        /DNOMINMAX                  # NOMINMAX is required in order to play nice with std::min/std::max (otherwise Windows.h defines it's own)
        /wd4201 /wd4251
    )
endif()

set_target_properties(hailort-LightNet PROPERTIES CXX_STANDARD 14)
