"""图表生成工具。

通过 Kroki API 将 Mermaid / PlantUML / Graphviz / D2 等图表代码渲染为 SVG/PNG 图片，
零额外依赖，仅需 HTTP 请求。
"""

import os
import urllib.parse
import requests
from langchain.tools import tool

_KROKI_BASE = "https://kroki.io"
_FETCH_TIMEOUT = 15
_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "diagrams")

# Kroki 支持的全部图表类型 → 常用别名
_TYPE_ALIASES = {
    "mermaid": "mermaid",
    "mmd": "mermaid",
    "plantuml": "plantuml",
    "puml": "plantuml",
    "graphviz": "graphviz",
    "dot": "graphviz",
    "d2": "d2",
    "blockdiag": "blockdiag",
    "seqdiag": "seqdiag",
    "actdiag": "actdiag",
    "nwdiag": "nwdiag",
    "c4plantuml": "c4plantuml",
    "c4": "c4plantuml",
    "erd": "erd",
    "nomnoml": "nomnoml",
    "pikchr": "pikchr",
    "structurizr": "structurizr",
    "vega": "vega",
    "vegalite": "vegalite",
    "wavedrom": "wavedrom",
    "excalidraw": "excalidraw",
}


def _render_via_kroki(code: str, diagram_type: str, output_format: str) -> tuple[bytes, str]:
    """通过 Kroki API 渲染图表，返回 (数据, 错误信息)。"""
    request_url = f"{_KROKI_BASE}/{diagram_type}/{output_format}"
    try:
        resp = requests.post(
            request_url,
            data=code.encode("utf-8"),
            headers={"Content-Type": "text/plain; charset=utf-8"},
            timeout=_FETCH_TIMEOUT,
        )
        if resp.status_code == 200:
            return resp.content, ""
        elif resp.status_code == 400:
            return b"", f"Kroki 返回 400：图表语法可能有误，请检查代码。\n响应: {resp.text[:500]}"
        elif resp.status_code == 500:
            return b"", f"Kroki 渲染失败（500）：图表代码可能包含不支持的特性。"
        else:
            return b"", f"Kroki 返回 HTTP {resp.status_code}: {resp.text[:300]}"
    except requests.Timeout:
        return b"", f"请求超时（{_FETCH_TIMEOUT}s）"
    except requests.ConnectionError:
        return b"", "无法连接到 Kroki 服务（https://kroki.io）"
    except requests.RequestException as e:
        return b"", f"请求失败: {e}"


def _render_via_mermaid_ink(code: str, output_format: str) -> tuple[bytes, str]:
    """备用方案：通过 Mermaid.ink API 渲染。"""
    if output_format == "svg":
        url = "https://mermaid.ink/svg/"
    else:
        url = "https://mermaid.ink/img/"
    # Mermaid.ink 使用 base64url(pako_deflate) 编码
    import base64
    import zlib
    compressed = zlib.compress(code.encode("utf-8"), 9)
    encoded = base64.urlsafe_b64encode(compressed).rstrip(b"=").decode("ascii")
    try:
        resp = requests.get(url + encoded, timeout=_FETCH_TIMEOUT)
        if resp.status_code == 200:
            return resp.content, ""
        return b"", f"Mermaid.ink 返回 HTTP {resp.status_code}"
    except Exception as e:
        return b"", str(e)


@tool
def render_diagram(code: str, diagram_type: str = "mermaid", output_format: str = "svg") -> str:
    """将图表代码渲染为图片文件，保存到 diagrams/ 目录。

    支持多种图表类型：Mermaid（流程图、时序图、甘特图等）、PlantUML、Graphviz、D2 等。
    渲染结果保存为文件，返回文件路径供后续引用。

    Args:
        code: 图表源代码（Mermaid/PlantUML/Graphviz/D2 等格式）。
        diagram_type: 图表类型，常用值：mermaid, plantuml, graphviz, d2, c4plantuml, erd, nomnoml。
        output_format: 输出格式，svg 或 png（默认 svg）。

    常见 Mermaid 示例:
        流程图: flowchart TD\n  A[开始] --> B{判断}\n  B -->|是| C[执行]\n  B -->|否| D[结束]
        时序图: sequenceDiagram\n  Alice->>Bob: 请求\n  Bob-->>Alice: 响应
        甘特图: gantt\n  title 项目计划\n  section 开发\n  编码: 2024-01-01,30d
        类图: classDiagram\n  class Animal {\n    +name: str\n    +eat(): void\n  }
    """
    # 类型别名解析
    diagram_type = _TYPE_ALIASES.get(diagram_type.lower(), diagram_type.lower())
    if output_format not in ("svg", "png"):
        return "Error: output_format 仅支持 svg 或 png"

    if not code or len(code.strip()) < 5:
        return "Error: 图表代码太短（至少 5 个字符）"

    # 确保输出目录存在
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    # 生成唯一文件名
    import hashlib
    import time
    code_hash = hashlib.md5(code.encode()).hexdigest()[:8]
    filename = f"{diagram_type}_{code_hash}.{output_format}"
    filepath = os.path.join(_OUTPUT_DIR, filename)

    # 如果已存在同样的文件，直接返回（幂等）
    img_url = f"/diagrams/{filename}"
    if os.path.exists(filepath):
        return f"![图表]({img_url})\n\n图表已存在（相同代码已渲染过）: {filepath}"

    # 主方案：Kroki API
    data, error = _render_via_kroki(code, diagram_type, output_format)

    # 备用方案：Mermaid.ink（仅 Mermaid 类型）
    if error and diagram_type == "mermaid":
        data, error2 = _render_via_mermaid_ink(code, output_format)
        if not error2:
            error = ""

    if error:
        return f"Error: {error}"

    try:
        with open(filepath, "wb") as f:
            f.write(data)
        return f"![图表]({img_url})"
    except OSError as e:
        return f"Error: 保存文件失败 - {e}"
