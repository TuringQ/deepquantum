本目录用于构建 API 文档

## 依赖

Python >= 3.8

`pip install -r sphinx_doc/requirements.txt`


## 构建

`make html`


## 构建用于 AI 教育产品中的算法库文档的镜像

    docker build -t 192.168.20.55/ai_edu/algo-doc:v0.1.5 .
