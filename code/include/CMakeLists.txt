add_library(
    read_infile STATIC
    read_infile.hpp
    ../modules/read_infile.cpp
) 
set_target_properties(read_infile PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(read_infile PUBLIC ${YAML_CPP_LIBRARIES})
target_include_directories(read_infile PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")