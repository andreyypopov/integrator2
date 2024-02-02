CMAKE_MINIMUM_REQUIRED(VERSION 3.18)

include(CMakePrintHelpers)

PROJECT(integrator2 DESCRIPTION "The Integrator 2.0 library" LANGUAGES CXX CUDA)
SET(PROJECT_NAME "integrator2")
SET(PROJECT_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

INCLUDE(${PROJECT_ROOT_PATH}/cmake/functions.cmake)

SET(PROJECT_SOURCE_PATH ${PROJECT_ROOT_PATH}/src)

file(GLOB src_files
	${PROJECT_SOURCE_PATH}/common/cuda_math.cu
	${PROJECT_SOURCE_PATH}/Mesh3d.cu
	${PROJECT_SOURCE_PATH}/NumericalIntegrator3d.cu
	${PROJECT_SOURCE_PATH}/evaluators/evaluator3d.cu
	${PROJECT_SOURCE_PATH}/evaluators/evaluatorJ3DK.cu
	)
file(GLOB header_files
	${PROJECT_SOURCE_PATH}/common/constants.h
	${PROJECT_SOURCE_PATH}/common/cuda_helper.cuh
	${PROJECT_SOURCE_PATH}/common/cuda_math.cuh
	${PROJECT_SOURCE_PATH}/common/cuda_memory.cuh
	${PROJECT_SOURCE_PATH}/common/device_vector.cuh
	${PROJECT_SOURCE_PATH}/Mesh3d.cuh
	${PROJECT_SOURCE_PATH}/NumericalIntegrator3d.cuh
	${PROJECT_SOURCE_PATH}/evaluators/evaluator3d.cuh
	${PROJECT_SOURCE_PATH}/evaluators/evaluatorJ3DK.cuh
	)


if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
	set(CMAKE_INSTALL_PREFIX "${PROJECT_ROOT_PATH}/install" CACHE PATH "..." FORCE)
endif()

add_library(integrator2 SHARED)

setup_cuda(${PROJECT_NAME})

target_sources(integrator2 PUBLIC ${src_files})

add_subdirectory(tests/integrator3D)

install(TARGETS integrator2)
install(FILES ${header_files} DESTINATION include)

set_target_properties(integrator2 PROPERTIES VERSION 2.0)
set_target_properties(integrator2 PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

if(NOT DEFINED CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "RELWITHDEBINFO")
endif()

MESSAGE("The ${PROJECT_NAME} library will be installed to ${CMAKE_INSTALL_PREFIX}")
cmake_print_variables(CMAKE_BUILD_TYPE)
