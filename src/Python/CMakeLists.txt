cmake_minimum_required (VERSION 3.0)

project(tomobarPython)

# The version number.
message("Creating Python Wrapper")
# conda orchestrated build
message("tomobar_VERSION: ${tomobar_VERSION}")
#include (GenerateExportHeader)

find_package(PythonInterp REQUIRED)
if (PYTHONINTERP_FOUND)
  message ("Current Python " ${PYTHON_VERSION_STRING} " found " ${PYTHON_EXECUTABLE})
endif()

	
message("CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")
set(CMAKE_BUILD_TYPE "Release")

if(WIN32)
  set (FLAGS "/DWIN32 /EHsc /openmp ")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:MSVCRT.lib")
  
  set (EXTRA_LIBRARIES)
		
  message("library lib: ${LIBRARY_LIB}")
  
elseif(UNIX)
   set (FLAGS "-fopenmp -O2 -funsigned-char -Wall  -Wl,--no-undefined -std=c++0x")  
   set (EXTRA_LIBRARIES 
		"gomp"
		)
endif()

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/setup-tomobar.py.in" "${CMAKE_CURRENT_BINARY_DIR}/setup-tomobar.py.in")

find_package(PythonInterp)
find_package(PythonLibs)
if (PYTHONINTERP_FOUND)
  message(STATUS "Found PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}")
  message(STATUS "Python version ${PYTHON_VERSION_STRING}")
endif()
if (PYTHONLIBS_FOUND)
  message(STATUS "Found PYTHON_INCLUDE_DIRS=${PYTHON_INCLUDE_DIRS}")
  message(STATUS "Found PYTHON_LIBRARIES=${PYTHON_LIBRARIES}")
endif()

if (PYTHONINTERP_FOUND)
    message("Python found " ${PYTHON_EXECUTABLE})
    set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup-tomobar.py.in")
    set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup-tomobar.py")
    set (DEPS       "${CMAKE_BINARY_DIR}/src/Core/")
    set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp")

    configure_file(${SETUP_PY_IN} ${SETUP_PY})

    message("Core binary dir " ${CMAKE_BINARY_DIR}/Core/${CMAKE_BUILD_TYPE})
    
    if (CONDA_BUILD)
      add_custom_command(OUTPUT ${OUTPUT}
                       COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR}/src
                       COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/tomobar ${CMAKE_CURRENT_BINARY_DIR}/tomobar
                       COMMAND ${CMAKE_COMMAND} -E env tomobar_VERSION=${tomobar_VERSION}
                                                       PREFIX=${CMAKE_SOURCE_DIR}/src/Core
                                                       LIBRARY_INC=${CMAKE_SOURCE_DIR}/src/Core
                                                       LIBRARY_LIB=${CMAKE_BINARY_DIR}/src/Core
                                                       ${PYTHON_EXECUTABLE} ${SETUP_PY} install
                       COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
                       DEPENDS tomobar)

    else()
      if (WIN32)
        add_custom_command(OUTPUT ${OUTPUT}
                       COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR}/src
                       COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/tomobar ${CMAKE_CURRENT_BINARY_DIR}/tomobar
                       COMMAND ${CMAKE_COMMAND} -E env tomobar_VERSION=${tomobar_VERSION}
                                                       PREFIX=${CMAKE_SOURCE_DIR}/src/Core
                                                       LIBRARY_INC=${CMAKE_SOURCE_DIR}/src/Core
                                                       LIBRARY_LIB=${CMAKE_BINARY_DIR}/src/Core/${CMAKE_BUILD_TYPE}
                                                       ${PYTHON_EXECUTABLE} ${SETUP_PY} build_ext --inplace
                       COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
                       DEPENDS tomobar)
      else()
        add_custom_command(OUTPUT ${OUTPUT}
                       COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR}/src
                       COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/tomobar ${CMAKE_CURRENT_BINARY_DIR}/tomobar
                       COMMAND ${CMAKE_COMMAND} -E env tomobar_VERSION=${tomobar_VERSION}
                                                       PREFIX=${CMAKE_SOURCE_DIR}/src/Core
                                                       LIBRARY_INC=${CMAKE_SOURCE_DIR}/src/Core
                                                       LIBRARY_LIB=${CMAKE_BINARY_DIR}/src/Core
                                                       ${PYTHON_EXECUTABLE} ${SETUP_PY} build_ext --inplace
                       COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
                       DEPENDS tomobar)
      endif()
      install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tomobar 
              DESTINATION ${PYTHON_DEST})
    endif()
    
    
    add_custom_target(PythonWrapper ALL DEPENDS ${OUTPUT})

    #install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY} install)")
endif()
