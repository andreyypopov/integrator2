CMAKE_MINIMUM_REQUIRED(VERSION 3.18)

include(CMakePrintHelpers)

SET(PROJECT_NAME "integrator2test3D")
SET(PROJECT_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

INCLUDE(${PROJECT_ROOT_PATH}/../../cmake/functions.cmake)

if(NOT DEFINED INTEGRATOR_LIB_DIR)
    SET(INTEGRATOR_LIB_DIR "${PROJECT_ROOT_PATH}/../../install")
endif()

add_executable(${PROJECT_NAME} main.cu)
setup_cuda(${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} PRIVATE ${INTEGRATOR_LIB_DIR}/include)
target_link_directories(${PROJECT_NAME} PRIVATE ${INTEGRATOR_LIB_DIR}/lib)
target_link_libraries(${PROJECT_NAME} PRIVATE integrator2)

install(TARGETS ${PROJECT_NAME} DESTINATION bin)
