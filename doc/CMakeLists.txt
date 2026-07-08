# doc/CMakeLists.txt
find_package(Doxygen REQUIRED)
find_program(SPHINX_BUILD_EXE 
  NAMES sphinx-build
  HINTS 
    "$ENV{VIRTUAL_ENV}/bin" 
    "${CMAKE_SOURCE_DIR}/.venv/bin"
  NO_DEFAULT_PATH
)

if(NOT SPHINX_BUILD_EXE)
  message(FATAL_ERROR "sphinx-build executable not found! Please install Sphinx.")
endif()

set(DOXYGEN_GENERATE_HTML NO)
set(DOXYGEN_GENERATE_XML YES)
set(DOXYGEN_XML_OUTPUT "xml") 
set(DOXYGEN_EXCLUDE_SYMBOLS "INSTANTIATE" "stbi_*")

doxygen_add_docs(doxygen_xml
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src
  COMMENT "Generating XML documentation via Doxygen"
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/conf.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/conf.py
  @ONLY
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/index.rst
  ${CMAKE_CURRENT_BINARY_DIR}/index.rst
  COPYONLY
)
add_custom_target(docs
  COMMAND ${CMAKE_COMMAND} -E env VIRTUAL_ENV=${CMAKE_SOURCE_DIR}/.venv
    ${SPHINX_BUILD_EXE} -b html . ./html
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS doxygen_xml
  COMMENT "Building final HTML documentation with Sphinx + Breathe"
)