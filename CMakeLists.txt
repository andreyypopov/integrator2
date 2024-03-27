CMAKE_MINIMUM_REQUIRED(VERSION 3.18)

include(CMakePrintHelpers)

PROJECT(integrator2 DESCRIPTION "The Integrator 2.0 library" LANGUAGES CXX CUDA)
SET(PROJECT_NAME "integrator2")
SET(PROJECT_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

option(BUILD_TESTS "Build 3D test application" ON)

if (MSVC)
	SET(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

INCLUDE(${PROJECT_ROOT_PATH}/cmake/functions.cmake)

SET(PROJECT_SOURCE_PATH ${PROJECT_ROOT_PATH}/src)

SET(src_files
	${PROJECT_SOURCE_PATH}/Mesh3d.cu
	${PROJECT_SOURCE_PATH}/NumericalIntegrator3d.cu
	)
SET(src_common_files
	${PROJECT_SOURCE_PATH}/common/cuda_helper.cu
	${PROJECT_SOURCE_PATH}/common/cuda_math.cu
	)
SET(src_evaluators_files
	${PROJECT_SOURCE_PATH}/evaluators/evaluator3d.cu
	${PROJECT_SOURCE_PATH}/evaluators/evaluatorJ3DK.cu
	)

SET(header_files
	${PROJECT_SOURCE_PATH}/Mesh3d.cuh
	${PROJECT_SOURCE_PATH}/NumericalIntegrator3d.cuh
	${PROJECT_SOURCE_PATH}/QuadratureFormula3d.cuh
	)
SET(header_common_files
	${PROJECT_SOURCE_PATH}/common/constants.h
	${PROJECT_SOURCE_PATH}/common/cuda_helper.cuh
	${PROJECT_SOURCE_PATH}/common/cuda_math.cuh
	${PROJECT_SOURCE_PATH}/common/cuda_memory.cuh
	${PROJECT_SOURCE_PATH}/common/device_vector.cuh
	${PROJECT_SOURCE_PATH}/common/gpu_timer.cuh
	)
SET(header_evaluators_files
	${PROJECT_SOURCE_PATH}/evaluators/evaluator3d.cuh
	${PROJECT_SOURCE_PATH}/evaluators/evaluatorJ3DK.cuh
	)

source_group("library" FILES
	${src_files}
	${header_files}
)

source_group("library/common" FILES
	${src_common_files}
	${header_common_files}
)

source_group("library/evaluators" FILES
	${src_evaluators_files}
	${header_evaluators_files}
)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
	set(CMAKE_INSTALL_PREFIX "${PROJECT_ROOT_PATH}/install" CACHE PATH "Library installation path" FORCE)
endif()

add_library(integrator2 SHARED ${src_files} ${src_common_files} ${src_evaluators_files}
	${header_files} ${header_common_files} ${header_evaluators_files})

setup_cuda(${PROJECT_NAME})

if(BUILD_TESTS)
	add_subdirectory(tests/integrator3D)
	
	if (MSVC)
		set_property(DIRECTORY ${PROJECT_ROOT_PATH} PROPERTY VS_STARTUP_PROJECT integrator2test3D)
	endif()
endif()

install(TARGETS integrator2)
install(FILES ${header_files} DESTINATION include)
install(FILES ${header_common_files} DESTINATION include/common)
install(FILES ${header_evaluators_files} DESTINATION include/evaluators)

set_target_properties(integrator2 PROPERTIES VERSION 2.0)
set_target_properties(integrator2 PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

SET(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo" CACHE STRING "Available build configurations" FORCE)
if(NOT CMAKE_BUILD_TYPE)
	SET(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build configuration (available options: ${CMAKE_CONFIGURATION_TYPES})" FORCE)
endif()

MESSAGE("The ${PROJECT_NAME} library will be installed to ${CMAKE_INSTALL_PREFIX}")
cmake_print_variables(CMAKE_BUILD_TYPE)
