# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

import deepquantum as dq

sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'DeepQuantum'
copyright = '2026, TuringQ'
author = 'TuringQ'
release = f'{dq.__version__}'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # 自动生成 API 文档
    'sphinx.ext.napoleon',  # 支持 NumPy/Google 风格注释
    'sphinx.ext.viewcode',  # 增加“查看源码”链接
    'sphinx.ext.mathjax',  # 数学公式渲染
    'sphinx.ext.autosummary',  # 自动生成模块摘要
    'myst_nb',  # 核心：解析 Notebook 和 MyST Markdown
]

add_module_names = False
autosummary_generate = True

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

nb_execution_mode = 'off'

# MyST 语法扩展（开启后支持更多类似 LaTeX 的高级语法）
myst_enable_extensions = [
    'amsmath',
    'dollarmath',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'en'
# language = 'zh_CN'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_title = f'{project} v{release}'

html_theme_options = {
    # 启用右侧的“在 GitHub 上编辑/查看”按钮
    'repository_url': 'https://github.com/TuringQ/deepquantum',  # 改成你的仓库地址
    'use_repository_button': True,  # 开启 GitHub 仓库链接
    'use_download_button': True,  # 开启当前页面(ipynb)下载按钮
    'use_fullscreen_button': True,  # 开启全屏阅读按钮
    # 左侧导航栏配置
    'home_page_in_toc': False,
    'show_navbar_depth': 3,
    'collapse_navigation': False,
}
