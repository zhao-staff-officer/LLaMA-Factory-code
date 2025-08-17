# LLaMA-Factory 大模型微调
# 介绍

LLaMA-Factory 是一个专为开发者和研究人员设计的开源大语言模型（LLM）全栈开发框架，旨在通过整合前沿技术与友好工具链，大幅降低模型微调和部署的技术门槛。其核心作用体现在以下多个维度：

## 一、高效微调的「一站式工厂」

作为统一框架，LLaMA-Factory 集成了 **100+ 主流开源模型**（包括 LLaMA、Qwen、Baichuan、ChatGLM 等）的适配能力，并提供 **32 比特全参微调、16 比特冻结微调、LoRA、QLoRA 等 10+ 训练策略**。通过内置的 **LlamaBoard Web 界面**，用户无需编写代码即可完成从数据集构建到训练参数配置的全流程操作，例如将自定义数据集（如商品文案生成数据）快速转换为符合 Alpaca/ShareGPT 格式的训练样本，并实时监控损失曲线和显存占用。框架还支持 **混合精度训练**（BF16/FP16）和 **FlashAttention-2/S2 Attention 加速**，显著提升训练效率，例如在 RTX 4090 上可流畅微调 8B 级模型。

## 二、模型优化与部署的「智能工坊」

LLaMA-Factory 提供 **2-8 比特量化技术**（如 GPTQ、AWQ、QLoRA），可将模型体积压缩 50% 以上，同时保持 95%+ 的推理精度，适配边缘设备和低资源环境。其 **多阶段训练管线**支持从监督微调（SFT）到基于人类反馈的强化学习（RLHF）、直接偏好优化（DPO）的全链路对齐，例如通过奖励模型训练提升模型在特定领域（如医疗问答）的回答质量。此外，框架内置 **vLLM 和 Transformers 双推理引擎**，并支持导出为 Ollama、Hugging Face 等主流格式，方便集成到生产系统。

## 三、开发者赋能的「资源枢纽」

框架提供 **开箱即用的工具集**，包括英文文档的 AI 翻译、大模型调用代码生成、训练命令转 VSCode 调试配置等实用功能llamafactory.cn。通过 **数据工厂模块**，用户可快速构建自定义数据集（如将弱智吧对话数据转换为训练样本），并利用内置的 **50+ 标准数据集模板**（如 Alpaca、ShareGPT）加速开发。社区还维护了 **高质量 Prompts 库**和 **技术文章聚合平台**，帮助开发者掌握提示工程技巧和行业前沿动态llamafactory.cn。

## 四、多语言与多模态的「全球化适配」

LLaMA-Factory 深度支持中文等多语言场景，例如通过 **RoPE 缩放技术**优化长文本处理能力，并提供 **中文数据集标注工具**和 **双语模型训练示例**。在多模态领域，框架已支持 LLaVA 等模型的视觉 - 语言联合训练，可实现图文对齐任务的高效微调。其 **分布式训练架构**（如 DeepSpeed ZeRO-3）允许在多 GPU 集群上扩展训练规模，满足企业级应用需求。

通过将复杂的训练流程抽象为模块化操作，LLaMA-Factory 让开发者专注于业务逻辑创新，而非底层工程实现。无论是学术研究中的模型迭代，还是工业级 AI 产品的快速落地，该框架均能提供从数据到部署的完整技术链路支持，成为推动大模型应用民主化的重要基础设施。





# 项目说明

#### [官方文档](https://github.com/hiyouga/LLaMA-Factory/blob/main/README_zh.md )

#### [版本0.9.3](https://codeload.github.com/hiyouga/LLaMA-Factory/zip/refs/tags/v0.9.3)

#### 环境

##### 	python版本：3.10 

##### 	pytourch版本：12.6

##### 	基础模型：DeepSeek-R1-Distill-Qwen-1.5B



#### 基础资料

#####  	[CUDA参考资料](https://cloud.tencent.com/developer/article/2089949) 

##### 	[HuggingFace模型下载地址](https://huggingface.co/)

##### 	[HF-Mirror模型下载加速地址](https://hf-mirror.com/)



















