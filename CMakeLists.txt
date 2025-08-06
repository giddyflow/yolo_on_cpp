cmake_minimum_required(VERSION 3.16)
project(YoloLiveDetector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EXECUTABLE_NAME YoloDemo)

set(DATA_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/yolov4-tiny.cfg    # <- ИЗМЕНЕНО
    ${CMAKE_CURRENT_SOURCE_DIR}/yolov4-tiny.weights # <- ИЗМЕНЕНО
    ${CMAKE_CURRENT_SOURCE_DIR}/coco.names
)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose build type: Debug, Release, RelWithDebInfo, MinSizeRel" FORCE)
  message(STATUS "Build type not specified, defaulting to 'Release'.")
endif()

find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc videoio dnn)

message(STATUS "Found OpenCV ${OpenCV_VERSION}")
message(STATUS "OpenCV library status:")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "    libraries: ${OpenCV_LIBS}")

add_executable(${EXECUTABLE_NAME} src/main.cpp)

target_include_directories(${EXECUTABLE_NAME} PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(${EXECUTABLE_NAME} PUBLIC ${OpenCV_LIBS})

message(STATUS "Configuration successful. Executable '${EXECUTABLE_NAME}' will be built from 'src/main.cpp'.")