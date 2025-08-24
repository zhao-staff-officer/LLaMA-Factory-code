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

1. ##### 	python版本：3.10 

2. ##### 	pytourch版本：12.6

3. ##### 	基础模型：DeepSeek-R1-Distill-Qwen-1.5B



#### 基础资料

1. #####  	[CUDA参考资料](https://cloud.tencent.com/developer/article/2089949) 

2. ##### 	[HuggingFace模型下载地址](https://huggingface.co/)

3. ##### 	[HF-Mirror模型下载加速地址](https://hf-mirror.com/)



# 训练

## 命令

```shell
# 启动训练
# llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml 
```



## GPU运行状态

```shell
#查看GPU运行状态
#nvitop -m auto
```



## YAML备注

```yaml
### model
#适合可以链接网络下载模型的场景
#model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
#适合不支持联网下载的模型场景
model_name_or_path: E:\dataSource\LLModels\modelscope\Qwen3-0.6B
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

### dataset
#训练数据集
dataset: identity,alpaca_zh_demo
template: llama3
cutoff_len: 2048
max_samples: 100
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 0

### output
#保存地址
output_dir: saves/Qwen3-0.6B/lora/sft
#每50条记录一次
logging_steps: 50
#每50条保存一次
save_steps: 50
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none  # choices: [none, wandb, tensorboard, swanlab, mlflow]

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null

### eval
# eval_dataset: alpaca_en_demo
# val_size: 0.1
# per_device_eval_batch_size: 1
# eval_strategy: steps
# eval_steps: 500

```

# 推理

## 命令

```shell
# 原始模型推理
# llamafactory-cli chat examples/inference/llama3.yaml

# 训练模型推理
# llamafactory-cli chat examples/inference/llama3_lora_sft.yaml
```



## 原始模型YAML备注

```yaml
#原始模型推理
model_name_or_path: E:\dataSource\LLModels\modelscope\Qwen3-0.6B
template: llama3
infer_backend: huggingface  # choices: [huggingface, vllm, sglang]
trust_remote_code: true

```





## 训练模型YAML备注

```yaml
#模型地址
model_name_or_path: E:\dataSource\LLModels\modelscope\Qwen3-0.6B
#微调保存地址
adapter_name_or_path: saves/Qwen3-0.6B/lora/sft
template: llama3
infer_backend: huggingface  # choices: [huggingface, vllm, sglang]
trust_remote_code: true

```





# 合并

## 命令

```shell
# 导出模型
# llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```



## YAML备注

```yaml
### Note: DO NOT use quantized model or quantization_bit when merging lora adapters

### model
#合并模型路径
model_name_or_path: E:\dataSource\LLModels\modelscope\Qwen3-0.6B
#训练结果路径
adapter_name_or_path: saves/Qwen3-0.6B/lora/sft
template: llama3
trust_remote_code: true

### export
#导出路径
export_dir: output/Qwen3-0.6B-lora_sft
export_size: 5
export_device: cpu  # choices: [cpu, auto]
export_legacy_format: false
```



# 量化



## PTQ

后训练量化（PTQ, Post-Training Quantization）一般是指在模型预训练完成后，基于校准数据集（calibration dataset）确定量化参数进而对模型进行量化。

## GPTQ

GPTQ(Group-wise Precision Tuning Quantization)是一种静态的后训练量化技术。”静态”指的是预训练模型一旦确定,经过量化后量化参数不再更改。GPTQ 量化技术将 fp16 精度的模型量化为 4-bit ,在节省了约 75% 的显存的同时大幅提高了推理速度。 为了使用GPTQ量化模型，您需要指定量化模型名称或路径，例如 `model_name_or_path: TechxGenus/Meta-Llama-3-8B-Instruct-GPTQ`

## QAT

在训练感知量化（QAT, Quantization-Aware Training）中，模型一般在预训练过程中被量化，然后又在训练数据上再次微调，得到最后的量化模型。

## AWQ

AWQ（Activation-Aware Layer Quantization）是一种静态的后训练量化技术。其思想基于：有很小一部分的权重十分重要，为了保持性能这些权重不会被量化。 AWQ 的优势在于其需要的校准数据集更小，且在指令微调和多模态模型上表现良好。 为了使用 AWQ 量化模型,您需要指定量化模型名称或路径，例如 `model_name_or_path: TechxGenus/Meta-Llama-3-8B-Instruct-AWQ`

## AQLM

AQLM（Additive Quantization of Language Models）作为一种只对模型权重进行量化的PTQ方法，在 2-bit 量化下达到了当时的最佳表现，并且在 3-bit 和 4-bit 量化下也展示了性能的提升。 尽管 AQLM 在模型推理速度方面的提升并不是最显著的，但其在 2-bit 量化下的优异表现意味着您可以以极低的显存占用来部署大模型。

## OFTQ

OFTQ(On-the-fly Quantization)指的是模型无需校准数据集，直接在推理阶段进行量化。OFTQ是一种动态的后训练量化技术. OFTQ在保持性能的同时。 因此，在使用OFTQ量化方法时，您需要指定预训练模型、指定量化方法 `quantization_method` 和指定量化位数 `quantization_bit` 下面提供了一个使用bitsandbytes量化方法的配置示例：

```
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
quantization_bit: 4
quantization_method: bitsandbytes  # choices: [bitsandbytes (4/8), hqq (2/3/4/5/6/8), eetq (8)]
```

## bitsandbytes

区别于 GPTQ, bitsandbytes 是一种动态的后训练量化技术。bitsandbytes 使得大于 1B 的语言模型也能在 8-bit 量化后不过多地损失性能。 经过bitsandbytes 8-bit 量化的模型能够在保持性能的情况下节省约50%的显存。

## HQQ

依赖校准数据集的方法往往准确度较高，不依赖校准数据集的方法往往速度较快。HQQ（Half-Quadratic Quantization）希望能在准确度和速度之间取得较好的平衡。作为一种动态的后训练量化方法，HQQ无需校准阶段， 但能够取得与需要校准数据集的方法相当的准确度，并且有着极快的推理速度。

## EETQ

EETQ(Easy and Efficient Quantization for Transformers)是一种只对模型权重进行量化的PTQ方法。具有较快的速度和简单易用的特性。





## YAML备注



| 参数名称                | 类型                                   | 介绍                                                         | 默认值       |
| ----------------------- | -------------------------------------- | ------------------------------------------------------------ | ------------ |
| quantization_method     | Literal[“bitsandbytes”, “hqq”, “eetq”] | 指定用于量化的算法，支持 “bitsandbytes”, “hqq” 和 “eetq”。   | bitsandbytes |
| quantization_bit        | Optional[int]                          | 指定在量化过程中使用的位数，通常是4位、8位等。               | None         |
| quantization_type       | Literal[“fp4”, “nf4”]                  | 量化时使用的数据类型，支持 “fp4” 和 “nf4”。                  | nf4          |
| double_quantization     | bool                                   | 是否在量化过程中使用 double quantization，通常用于 “bitsandbytes” int4 量化训练。 | True         |
| quantization_device_map | Optional[Literal[“auto”]]              | 用于推理 4-bit 量化模型的设备映射。需要 “bitsandbytes >= 0.43.0”。 | None         |
