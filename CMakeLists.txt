cmake_minimum_required(VERSION 2.8.4)
project(spookfish)

set(SOURCE_FILES main.cpp)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

configure_file(raytracing.cl raytracing.cl COPYONLY)
configure_file(transformation.cl transformation.cl COPYONLY)
configure_file(slicing.cl slicing.cl COPYONLY)
configure_file(shader.vert shader.vert COPYONLY)
configure_file(shader.frag shader.frag COPYONLY)

add_executable(spookfish ${SOURCE_FILES})
target_link_libraries(spookfish boost_program_options GL glut GLEW OpenCL dcmimgle dcmdata oflog ofstd pthread z )

add_definitions(-DUSE_STD_CXX_INCLUDES)
add_definitions(-DHAVE_CONFIG_H)
