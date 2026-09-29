# 新闻短视频生成流水线 · 简历素材

中英文两版项目条目，可直接粘贴到简历。文中的数字均可在代码中核对：
5 个流水线阶段、3 个本地模型服务、约 3.4k 行 Python、重写上限 3 轮、
110–185 词长度区间（`agent_core/config.py`）、ComfyUI 300s 就绪预算。

---

## 中文版（简历直接粘贴）

**新闻短视频生成流水线**（个人项目）

**技术栈**：Python、LangGraph、LangSmith、llama.cpp、Qwen3-TTS、ComfyUI（Z-Image-Turbo）、FastAPI、FFmpeg、pytest

**项目描述**：独立设计并实现的多模态内容生成系统。读取中文新闻原文，自动产出带英文配音的 1 分钟短视频：本地 LLM 生成播报稿并切分为单句，逐句生成图像提示词、配音与配图，最终按「一句一图一音」对齐合成为成片。全流程本地推理，无外部 API 依赖；以 LangGraph 将五个阶段编排为带自评反思循环的可观测工作流。

- **工作流编排**：将「撰写 → 自评 → 重写 → 媒体生成 → 合成」建模为 LangGraph 状态图，用条件边实现 critic 反思循环（编辑模型不通过则携带修改意见重写，上限 3 轮）；并在 LLM 自评之前先做确定性长度校验（110–185 词 ≈ 1 分钟），避免小模型不可靠的评判浪费整轮生成。
- **容错与降级**：节点内隔离异常，单句配音或绘图失败只记入 manifest 并跳过，其余句子照常出片；LLM 调用带重试与截断兜底，模型服务不可用时降级继续而非中断。
- **模型服务托管**：统一封装 llama.cpp、Qwen3-TTS、ComfyUI 三个本地服务的生命周期 —— 按需拉起、健康探针、按模型定制就绪预算（ComfyUI 冷启动 300s），并自动复用已在运行的服务实例。
- **自研 TTS 推理服务**：基于 FastAPI 实现「句子 + 语言 + 音色/情感 → WAV」的推理后端，从 checkpoint 的 `config.json` 自动识别模型类型，自适应音色克隆与 tone 描述控制，并通过 `/health` 提供能力协商。
- **音画对齐与测试**：以统一文件名词干贯穿文本、提示词、音频、图像四类产物，实现确定性的「一句一图一音」配对，FFmpeg 逐段渲染后拼接；pytest 离线端到端测试以假模型服务 + 真实 FFmpeg 覆盖全链路，不加载模型即可回归。

---

## English Version

**AI News Video Pipeline** — personal project

**Tech:** Python, LangGraph, LangSmith, llama.cpp, Qwen3-TTS, ComfyUI (Z-Image-Turbo), FastAPI, FFmpeg, pytest

A fully local multimodal pipeline that turns a Chinese news article into a one-minute English news video. A local LLM writes the script and splits it per sentence; each sentence then gets an image prompt, a TTS take and a generated illustration; FFmpeg pairs every image with its audio and concatenates the final cut. No external APIs — the five stages run as an observable LangGraph workflow with a critic-driven rewrite loop.

- **Workflow orchestration:** Modelled *write → review → rewrite → generate media → compose* as a LangGraph state machine; conditional edges implement the critic rewrite loop (max 3 rounds), guarded by a deterministic length check (110–185 words ≈ 1 minute) so an unreliable small model's self-review cannot waste a generation pass.
- **Failure isolation:** Nodes never raise — a failed TTS or image call is recorded in the manifest and skipped while the rest of the video still renders.
- **Model server lifecycle:** One manager starts, health-checks and reclaims three local servers (llama.cpp, Qwen3-TTS, ComfyUI) with per-model readiness budgets, reusing instances already running.
- **TTS inference service:** Built a FastAPI backend (*sentence + language + tone → WAV*) that detects the checkpoint type from its `config.json` and adapts to voice cloning or tone control, with `/health` capability negotiation.
- **Deterministic A/V pairing:** One shared filename stem across text, prompt, audio and image makes pairing exact; FFmpeg renders each segment then concatenates, falling back from stream copy to re-encode.

---

## 技术细节（面试可展开）

### 反思循环前的确定性闸门

本地 3B 模型的自评结果不稳定，直接以它作为重写依据会浪费整轮生成。因此先用确定性词数校验
（目标 150 词，110–185 词区间）拦截明显不合规的稿件，只有长度合格的稿件才交给编辑模型评判。
编辑模型不可用时接受当前稿件，不让整条流水线卡在评审环节。

### 定位并修复 FFmpeg 的静音缺陷

实测发现 `-shortest` 对 `-loop 1` 的静帧不生效：每段视频尾部会比音频多出约 1.9 秒静音
（3.0 秒音频渲染出 4.88 秒片段）。改为先用 ffprobe 探测音频时长、再以 `-t` 精确裁剪，
12 句的视频消除了约 23 秒静音。修复以可选参数的形式加在共享的 `step5.py:make_segment()` 上，
agent_core 工作流传入音频时长；step5 自身的 main() 仍按原样调用，行为不变。

### 不加载模型也能回归的测试策略

假 TTS 用 `wave` 模块写出真实 WAV，假绘图用 FFmpeg 生成真实 PNG，因此 x264 编码、
AAC 混音与 concat 拼接等渲染路径在测试中被真实执行，而不是被打桩绕过。
`pytest tests/test_agent_core.py` 全流程跑完约 7 秒，不需要启动任何模型服务。

---

## 项目结构速览

```
step1.py … step5.py     脚本式五步流程（可单独运行）
agent_core/             LangGraph 工作流：graph.py 编排，tools/ 放各阶段工具
ai/manage_model.py      三个本地模型服务的进程管理与 HTTP 客户端
ai/tts-backend/         自研 Qwen3-TTS FastAPI 推理服务
resource/               输入输出：text/ image_prompt/ audio/ images/ video/
tests/                  pytest 测试
```

运行：`python -m agent_core.main`
