# doc/conf.py.in

project = 'tunx'
copyright = '2026, Tung pham'
author = 'Tung pham'
release = '1.0'

extensions = [
  'sphinx.ext.autodoc',
  'breathe',
]

breathe_projects = {
  "tunx": "/home/unixaccount/workspace/tunx/doc/xml"
}
breathe_default_project = "tunx"

html_theme = 'sphinx_rtd_theme'
