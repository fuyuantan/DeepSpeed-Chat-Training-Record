# DeepSpeed-Chat-Training-Record
DeepSpeed-Chat from https://github.com/deepspeedai/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat/

<details>
<summary>Step1 SFT</summary>
    
## I. 参数设置

### 1. 模型与训练策略
*   **基础模型 (`model_name_or_path`)**: `facebook/opt-1.3b`
*   **LoRA 维度 (`lora_dim`)**: `128`
*   **梯度累积步数 (`gradient_accumulation_steps`)**: `16`

### 2. 分布式训练 (DeepSpeed Launcher)
*   **执行命令 (`cmd`)**:
    ```bash
    /root/miniconda3/bin/python -u -m deepspeed.launcher.launch --world_info=eyJsb2NhbGhvc3QiOiBbMF19 --master_addr=127.0.0.1 --master_port=29500 --enable_each_rank_log=None main.py --model_name_or_path facebook/opt-1.3b --gradient_accumulation_steps 16 --lora_dim 128 --zero_stage 0 --enable_tensorboard --tensorboard_path /root/DeepSpeedExamples/applications/DeepSpeed-Chat/output/actor-models/1.3b --deepspeed --output_dir /root/DeepSpeedExamples/applications/DeepSpeed-Chat/output/actor-models/1.3b
    ```
*   **节点信息 (`world_info`)**: `{'localhost': [0]}` (本地 GPU 0)
*   **主节点地址 (`master_addr`)**: `127.0.0.1`
*   **主节点端口 (`master_port`)**: `29500`
*   **节点数量 (`nnodes`)**: `1`
*   **本地进程数 (`num_local_procs`)**: `1` (使用1个GPU)
*   **总进程数/世界大小 (`dist_world_size`)**: `1`
*   **可见CUDA设备 (`CUDA_VISIBLE_DEVICES`)**: `0`

### 3. DeepSpeed 配置
#### a. 批处理大小
*   **训练总批次大小 (`train_batch_size`)**: `128`
*   **每个GPU的微批次大小 (`train_micro_batch_size_per_gpu`)**: `8`
    *   *(验证: 微批次 8 * 梯度累积 16 * GPU数量 1 = 128，与总批次大小相符)*

#### b. ZeRO 优化
*   **ZeRO 阶段 (`zero_stage` / `zero_optimization.stage`)**: `0`
*   **参数卸载 (`offload_param.device`)**: `none`
*   **优化器状态卸载 (`offload_optimizer.device`)**: `none`

#### c. 精度控制
*   **FP16 启用 (`fp16.enabled`)**: `True`
*   **FP16 损失缩放窗口 (`fp16.loss_scale_window`)**: `100`
*   **初始动态损失缩放值 (`initial_dynamic_scale` / `dynamic_loss_scale_args.init_scale`)**: `65536`

#### d. 优化器与学习率调度器
*   **使用的客户端优化器**: `FusedAdam`
*   **使用的客户端学习率调度器**: `torch.optim.lr_scheduler.LambdaLR`
*   **初始学习率 (`lr`)**: `0.001`
*   **初始动量 (`mom`)**: `(0.9, 0.95)`

#### e. 梯度处理
*   **梯度裁剪 (`gradient_clipping`)**: `1.0`
*   **梯度预缩放 (`prescale_gradients`)**: `false`

#### f. 日志与监控
*   **打印间隔步数 (`steps_per_print`)**: `10`
*   **TensorBoard 启用 (`tensorboard.enabled`)**: `True`
*   **TensorBoard 输出路径 (`tensorboard.output_path`)**: `/root/DeepSpeedExamples/applications/DeepSpeed-Chat/output/actor-models/1.3b/ds_tensorboard_logs/`
*   **TensorBoard 任务名称 (`tensorboard.job_name`)**: `step1_model_tensorboard`

#### g. 输出目录
*   **模型输出目录 (`output_dir`)**: `/root/DeepSpeedExamples/applications/DeepSpeed-Chat/output/actor-models/1.3b`

## II. 训练细节与执行过程

### 1. 环境与设置
*   **加速器 (`ds_accelerator`)**: `cuda` (自动检测)
*   **Python 环境**: `/root/miniconda3/bin/python`
*   **主机文件 (`hostfile`)**: 未找到，仅使用本地资源进行训练。
*   **CUDA 版本**: 安装版本 `11.8`，PyTorch 编译版本 `11.7` (被认为兼容)。
*   **DeepSpeed NCCL 后端**: 利用 PyTorch 的 NCCL 后端进行通信。
*   **FusedAdam 算子编译**: 耗时 `50.531` 秒。
*   **DeepSpeed 版本**: `0.9.5`

### 2. 训练运行
*   **训练轮次 (Epochs)**: `1`
*   **每轮的总微批次数 (Total Micro Batches)**: `1907`
*   **报告的模型参数量**: `1.429 B`
*   **序列长度 (Sequence Length)**: `512`

## III. 损失 (Loss) 与困惑度 (Perplexity) 变化

### 1. 初始评估 (Epoch 0/1)
*   **时间戳**: `[2025-05-02 10:52:48,xxx]`
*   **困惑度 (ppl)**: `8.376730918884277`
*   **损失 (loss)**: `2.125457763671875`

### 2. 训练过程中 (Epoch 1/1)
*   **性能指标 (代表性数值，存在波动)**:
    *   **延迟 (Latency)**: ~`0.33s` / micro-batch
    *   **TFLOPs**: ~`70-71`
    *   **Samples/sec**: ~`24`
    *   **Time/seq**: ~`0.04s`
    *   *注: 日志中出现过 `0.18s` 延迟和 `127.52` TFLOPs 的峰值，可能与批次大小或系统波动有关。*

### 3. 最终评估 (Epoch 1/1)
*   **时间戳**: `[2025-05-02 11:04:35,xxx]`
*   **困惑度 (ppl)**: `5.937998294830322`
*   **损失 (loss)**: `1.7813720703125`

## IV. 总结与变化

*   训练总共进行了 **1 个 epoch**。
*   **损失 (loss)** 从初始的 `2.125` 降低到 `1.781`。
*   **困惑度 (perplexity, ppl)** 从初始的 `8.377` 改善至 `5.938`。
*   主要训练循环和最终评估过程大约耗时: **11 分 47 秒** (不包括初始环境设置和 FusedAdam 编译时间)。

</details>


<details>
<summary>Step2 Reward Model</summary>

</details>


<details>
<summary>Step3 PPO-RLHF</summary>

</details>


