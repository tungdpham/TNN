# doc/conf.py.in

project = 'synet'
copyright = '2026, Tung pham'
author = 'Tung pham'
release = '1.0'

extensions = [
  'sphinx.ext.autodoc',
  'breathe',
]

breathe_projects = {
  "synet": "/home/unixaccount/workspace/synet/doc/xml"
}
breathe_default_project = "synet"

html_theme = 'sphinx_rtd_theme'
