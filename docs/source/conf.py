# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import re
import shutil
import sys

import deepquantum as dq

sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'DeepQuantum'
copyright = '2026, TuringQ'  # noqa: A001
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
autodoc_inherit_docstrings = False
autodoc_typehints = 'both'
autodoc_typehints_description_target = 'documented'
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
    'colon_fence',
    'dollarmath',
]

# Markdown 管道表会把 |0\rangle、矩阵里的 | 等误判为列分隔符，从而撕裂 dollarmath。
# 文档中如需表格请使用 MyST 的 {list-table}（或 HTML table）；勿依赖 | col | 语法。
# myst_disable_syntax = ['table']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'en'
# language = 'zh_CN'

# MathJax：tex-mml-chtml.js 是浏览器端公式渲染入口（TeX/MathML → HTML）。
# Sphinx 默认从 jsDelivr 加载，国内常出现 net::ERR_CONNECTION_TIMED_OUT（连接挂起约 1 分钟后失败）。
# 若存在自托管目录（见 docs/setup_mathjax_static.sh），优先使用；否则改用 unpkg。
_conf_dir = os.path.dirname(os.path.abspath(__file__))
_mathjax_local = os.path.join(_conf_dir, '_static', 'mathjax-es5', 'tex-mml-chtml.js')
if os.path.isfile(_mathjax_local):
    mathjax_path = 'mathjax-es5/tex-mml-chtml.js'
else:
    mathjax_path = 'https://unpkg.com/mathjax@3.2.2/es5/tex-mml-chtml.js'

# MathJax 3 配置：支持 \ket \bra 等量子力学常用宏（与上面 mathjax 3.2.x 路径一致）
mathjax3_config = {
    'tex': {
        'macros': {
            'ket': ['{\\left|#1\\right\\rangle}', 1],
            'bra': ['{\\left\\langle#1\\right|}', 1],
            'braket': ['{\\left\\langle#1\\middle|#2\\right\\rangle}', 2],
        }
    }
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_css_files = ['custom.css', 'site-chrome.css']
html_js_files = [
    'sidebar-viewport-sync.js',
    'sidebar-click-logger.js',
    'iframe-fullscreen-fallback.js',
    'site-chrome.js',
]
html_title = f'{project} v{release}'

# 文档始终使用浅色模式，不跟随系统 dark / auto（与 _templates/layout.html 中脚本一致）
html_context = {
    'default_mode': 'light',
}

html_theme_options = {
    # 启用右侧的“在 GitHub 上编辑/查看”按钮
    'repository_url': 'https://github.com/TuringQ/deepquantum',
    'use_repository_button': True,  # 开启 GitHub 仓库链接
    'use_download_button': True,  # 开启当前页面(ipynb)下载按钮
    'use_fullscreen_button': True,  # 开启全屏阅读按钮
    # 左侧导航栏配置
    'home_page_in_toc': False,
    # show_navbar_depth=0 使 Quick Start / Tutorials / Demos 等 caption 可折叠，显示展开/收起箭头
    'show_navbar_depth': 0,
    'collapse_navigation': False,
}


def _copy_demos_fig_folders(app, exception):
    """Copy 'images' directories from subfolders in 'docs/source/demos' to the HTML build directory."""
    if exception:
        return
    conf_dir = os.path.dirname(os.path.abspath(__file__))
    demos_src = os.path.join(conf_dir, 'demos')
    outdir_demos = os.path.join(app.outdir, 'demos')
    for root, dirs, _ in os.walk(demos_src):
        rel = os.path.relpath(root, demos_src)
        if 'images' in dirs:
            img_src = os.path.join(root, 'images')
            img_dst = os.path.join(outdir_demos, rel, 'images')
            os.makedirs(os.path.dirname(img_dst), exist_ok=True)
            if os.path.exists(img_dst):
                shutil.rmtree(img_dst)
            shutil.copytree(img_src, img_dst)


def _fix_cases_caption_in_html(app, exception):
    """修复 demos 等子页面侧边栏 caption 被主题改成小写的问题。

    pydata/sphinx-book 会从 docname 路径推导 caption；构建完成后统一改回 toctree 中的大小写。
    """
    if exception:
        return
    outdir = app.outdir
    replacements = (
        ('<span class="caption-text">cases</span>', '<span class="caption-text">Cases</span>'),
        ('<span class="caption-text">demos</span>', '<span class="caption-text">Demos</span>'),
    )
    for root, _dirs, files in os.walk(outdir):
        for f in files:
            if f.endswith('.html'):
                path = os.path.join(root, f)
                try:
                    with open(path, encoding='utf-8') as fp:
                        content = fp.read()
                    new_content = content
                    for old, new in replacements:
                        new_content = new_content.replace(old, new)
                    if new_content != content:
                        with open(path, 'w', encoding='utf-8') as fp:
                            fp.write(new_content)
                except (OSError, UnicodeDecodeError):
                    pass


def _preprocess_myst_for_notebook(md_content, md_src_dir):
    """预处理 MyST markdown，使其适合放入 Jupyter notebook。
    - 展开 {include} 指令，将引用文件内容内联
    - 移除 {toctree} 等 Sphinx 指令块
    - 移除 HTML 注释
    """
    result = []

    # 1. 展开 {include} 指令
    include_pat = re.compile(
        r'```\{include\}\s+([^\s\n]+)\s*(?:\n[^\n]*)?\s*```',
        re.MULTILINE,
    )

    def replace_include(m):
        path_part = m.group(1).strip()
        inc_path = os.path.normpath(os.path.join(md_src_dir, path_part))
        if os.path.isfile(inc_path):
            try:
                with open(inc_path, encoding='utf-8') as fp:
                    return fp.read()
            except (OSError, UnicodeDecodeError):
                pass
        return m.group(0)

    md_content = include_pat.sub(replace_include, md_content)

    # 2. 移除 {toctree}、{contents} 等 Sphinx 指令块（```{directive}...```）
    sphinx_directive_pat = re.compile(
        r'```\{[a-zA-Z_]+\}.*?```',
        re.DOTALL,
    )
    md_content = sphinx_directive_pat.sub('', md_content)

    # 3. 移除 HTML 注释
    md_content = re.sub(r'<!--.*?-->', '', md_content, flags=re.DOTALL)

    # 4. 清理多余空行
    md_content = re.sub(r'\n{3,}', '\n\n', md_content)
    return md_content.strip()


def _add_ipynb_download_for_md_pages(app, exception):
    """为 .md 源文件页面生成 .ipynb 并添加下载选项。"""
    if exception:
        return
    try:
        import nbformat
    except ImportError:
        return
    outdir = app.outdir
    srcdir = app.srcdir
    sources_dir = os.path.join(outdir, '_sources')
    for root, _dirs, files in os.walk(outdir):
        for f in files:
            if not f.endswith('.html'):
                continue
            full_path = os.path.join(root, f)
            rel_to_out = os.path.relpath(full_path, outdir)
            rel_dir = os.path.dirname(rel_to_out)
            name = os.path.splitext(f)[0]
            md_rel = (rel_dir + '/' + name + '.md') if rel_dir else (name + '.md')
            md_src = os.path.join(srcdir, md_rel.replace('/', os.sep))
            if not os.path.exists(md_src):
                continue
            ipynb_rel = (rel_dir + '/' + name + '.ipynb') if rel_dir else (name + '.ipynb')
            ipynb_dst = os.path.join(sources_dir, ipynb_rel.replace('/', os.sep))
            # 始终从 .md 重新生成 .ipynb，以应用 MyST 预处理
            try:
                with open(md_src, encoding='utf-8') as fp:
                    md_content = fp.read()
                md_src_dir = os.path.dirname(md_src)
                md_content = _preprocess_myst_for_notebook(md_content, md_src_dir)
                nb = nbformat.v4.new_notebook()
                nb.cells.append(nbformat.v4.new_markdown_cell(md_content))
                nb.cells[0].metadata['jupytext'] = {'formats': 'ipynb,md'}
                os.makedirs(os.path.dirname(ipynb_dst), exist_ok=True)
                with open(ipynb_dst, 'w', encoding='utf-8') as fp:
                    nbformat.write(nb, fp)
            except Exception:
                continue
            try:
                with open(full_path, encoding='utf-8') as fp:
                    content = fp.read()
                up_levels = rel_to_out.count(os.sep)  # 目录层级，用于构造相对路径
                prefix = ('../' * up_levels) if up_levels > 0 else './'
                sources_prefix = prefix + '_sources/'
                ipynb_href = sources_prefix + ipynb_rel.replace(os.sep, '/')
                ipynb_li = (
                    '      <li><a href="' + ipynb_href + '" download\n'
                    '   class="btn btn-sm btn-download-source-button dropdown-item"\n'
                    '   title="Download notebook"\n'
                    '   data-bs-placement="left" data-bs-toggle="tooltip"\n'
                    '>\n'
                    '  <span class="btn__icon-container">\n'
                    '  <i class="fas fa-file"></i>\n'
                    '  </span>\n'
                    '<span class="btn__text-container">.ipynb</span>\n'
                    '</a></li>\n'
                    '\n      \n'
                )
                if 'btn__text-container">.ipynb' in content:
                    esc_name = re.escape(name)
                    content = re.sub(
                        r'(<a href=")[^"]*' + esc_name + r'\.ipynb(" [^>]*>)',
                        r'\g<1>' + ipynb_href + r'\g<2>',
                        content,
                        count=1,
                    )
                else:
                    pat = re.compile(
                        r'(</li>\s*)(<li>\s*<button onclick="window\.print\(\)")',
                        re.DOTALL,
                    )

                    def repl(m):
                        return m.group(1) + ipynb_li + m.group(2)

                    if pat.search(content):
                        content = pat.sub(repl, content, count=1)

                with open(full_path, 'w', encoding='utf-8') as fp:
                    fp.write(content)
            except (OSError, UnicodeDecodeError):
                pass


def _normalize_sources_download_links(app, exception):
    """构建结束后统一处理：指向 _sources 的下载链接勿使用 target='_blank'，改为 download，在当前页触发下载。"""
    if exception:
        return
    outdir = app.outdir
    pat = re.compile(
        r'(<a\s+href="[^"]*_sources/[^"]+")\s+target="_blank"',
        re.MULTILINE,
    )
    for root, _dirs, files in os.walk(outdir):
        for f in files:
            if not f.endswith('.html'):
                continue
            full_path = os.path.join(root, f)
            try:
                with open(full_path, encoding='utf-8') as fp:
                    content = fp.read()
            except (OSError, UnicodeDecodeError):
                continue
            if 'target="_blank"' not in content or '_sources/' not in content:
                continue
            new_content = pat.sub(r'\1 download', content)
            if new_content != content:
                try:
                    with open(full_path, 'w', encoding='utf-8') as fp:
                        fp.write(new_content)
                except OSError:
                    pass


def _add_sidebar_hidden_for_demos_cases(app, exception):
    """demos 示例子页面在构建时为侧栏添加 pst-sidebar-hidden，避免刷新时先展开再收起的闪烁。"""
    if exception:
        return
    outdir = app.outdir
    for root, _dirs, files in os.walk(outdir):
        for f in files:
            if f.endswith('.html'):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, outdir).replace(os.sep, '/')
                if 'demos/' not in rel_path and 'cases/' not in rel_path:
                    continue
                try:
                    with open(full_path, encoding='utf-8') as fp:
                        content = fp.read()
                    # 若尚未包含 pst-sidebar-hidden，则添加到 #pst-primary-sidebar 的 class 中
                    if 'pst-sidebar-hidden' in content:
                        continue
                    old = '<div id="pst-primary-sidebar" class="bd-sidebar-primary bd-sidebar">'
                    new = '<div id="pst-primary-sidebar" class="bd-sidebar-primary bd-sidebar pst-sidebar-hidden">'
                    if old in content:
                        with open(full_path, 'w', encoding='utf-8') as fp:
                            fp.write(content.replace(old, new))
                except (OSError, UnicodeDecodeError):
                    pass


def setup(app):
    app.connect('build-finished', _copy_demos_fig_folders)
    app.connect('build-finished', _fix_cases_caption_in_html)
    app.connect('build-finished', _add_ipynb_download_for_md_pages)
    app.connect('build-finished', _normalize_sources_download_links)
    app.connect('build-finished', _add_sidebar_hidden_for_demos_cases)
