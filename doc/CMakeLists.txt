find_package(Doxygen REQUIRED)

include(FetchContent)

FetchContent_Declare(
  doxygen_awesome_theme
  GIT_REPOSITORY https://github.com/jothepro/doxygen-awesome-css.git
  GIT_TAG        v2.3.3
)
FetchContent_MakeAvailable(doxygen_awesome_theme)

FetchContent_GetProperties(doxygen_awesome_theme SOURCE_DIR THEME_DIR)

set(DOXYGEN_HTML_EXTRA_STYLESHEET "${THEME_DIR}/doxygen-awesome.css")

doxygen_add_docs(doxygen_docs
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src
  COMMENT "Generating API documentation with Doxygen Awesome theme"
)