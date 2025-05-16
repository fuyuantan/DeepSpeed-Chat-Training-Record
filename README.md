# DeepSpeed-Chat-Training-Record
DeepSpeed-Chat from https://github.com/deepspeedai/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat/

<details>
<summary>Step1 SFT</summary>
    
## I. 参数设置

### 1. 模型与训练策略
*   **基础模型 (`model_name_or_path`)**: `facebook/opt-1.3b`
*   **训练数据集**: `Dahoas/rm-static`，Total Micro Batches * train_micro_batch_size_per_gpu * world_size 1907 * 8 * 1 = 15256 个样本。数据集地址：https://huggingface.co/datasets/Dahoas/rm-static
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
# Reward Model (facebook/opt-350m) 训练日志分析

此日志详细记录了基于 `facebook/opt-350m` 的 Reward Model (RM) 的训练过程。

## I. 参数设置

### 1. 命令行参数
*   **基础模型 (`model_name_or_path`)**: `facebook/opt-350m`
*   **起始填充数 (`num_padding_at_beginning`)**: `1` (OPT模型特定参数)
*   **权重衰减 (`weight_decay`)**: `0.1`
*   **Dropout (`dropout`)**: `0.0` (显式设置，覆盖模型默认值)
*   **梯度累积步数 (`gradient_accumulation_steps`)**: `4`
*   **ZeRO 阶段 (`zero_stage`)**: `0`
*   **TensorBoard 启用**: `True`
    *   **路径 (`tensorboard_path`)**: `/root/DeepSpeedExamples/applications/DeepSpeed-Chat/output/reward-models/350m`
*   **输出目录 (`output_dir`)**: `/root/DeepSpeedExamples/applications/DeepSpeed-Chat/output/reward-models/350m`
*   **LoRA 维度 (`lora_dim`)**: 命令中未指定，默认为 `0` (未使用LoRA)。

### 2. 分布式训练 (DeepSpeed Launcher)
*   **节点信息 (`world_info`)**: `{'localhost': [0]}` (在本地机器的 GPU 0 上训练)
*   **主节点地址 (`master_addr`)**: `127.0.0.1`
*   **主节点端口 (`master_port`)**: `29500`
*   **节点数量 (`nnodes`)**: `1`
*   **本地进程数 (`num_local_procs`)**: `1` (使用1个GPU)
*   **分布式世界大小 (`dist_world_size`)**: `1`
*   **可见CUDA设备 (`CUDA_VISIBLE_DEVICES`)**: `0`

### 3. DeepSpeed 配置 (来自 JSON 和日志细节)
*   **批处理大小:**
    *   **每个GPU的训练微批次大小 (`train_micro_batch_size_per_gpu`)**: `8`
    *   **有效训练批次大小 (`train_batch_size`)**: `32`
        *   *(计算: 8 微批次/GPU * 1 GPU * 4 累积步数 = 32)*
*   **ZeRO 优化:**
    *   **阶段 (`zero_optimization.stage`)**: `0`
    *   参数卸载 (`offload_param.device`): `none`
    *   优化器卸载 (`offload_optimizer.device`): `none`
*   **精度:**
    *   **FP16 启用 (`fp16.enabled`)**: `True`
    *   损失缩放窗口 (`loss_scale_window`): `100`
    *   初始动态缩放 (`initial_dynamic_scale`): `65536`
*   **优化器与学习率调度器:**
    *   **使用的客户端优化器**: `FusedAdam`
    *   **使用的客户端学习率调度器**: `torch.optim.lr_scheduler.LambdaLR`
    *   **初始学习率 (`lr`)**: `[5e-05, 5e-05]`
    *   **初始动量 (`mom`)**: `[(0.9, 0.95), (0.9, 0.95)]`
*   **梯度:**
    *   **梯度裁剪 (`gradient_clipping`)**: `1.0`
*   **日志与 TensorBoard:**
    *   **打印间隔步数 (`steps_per_print`)**: `10`
    *   **TensorBoard 启用 (`tensorboard.enabled`)**: `True`
    *   **输出路径 (`tensorboard.output_path`)**: `/root/DeepSpeedExamples/applications/DeepSpeed-Chat/output/reward-models/350m/ds_tensorboard_logs/`
    *   **任务名称 (`tensorboard.job_name`)**: `step2_model_tensorboard`

## II. 训练细节与执行过程

### 1. 环境与设置
*   **加速器 (`ds_accelerator`)**: `cuda` (自动检测)
*   **主机文件**: 未找到，使用本地资源。
*   **Python 环境**: `/root/miniconda3/bin/python`
*   **TorchVision Beta 版本警告**: 常规警告。
*   **CUDA 版本**: 安装版本 `11.8`，PyTorch 编译版本 `11.7` (兼容)。
*   **DeepSpeed NCCL 后端**: 使用 PyTorch 的 NCCL。
*   **FusedAdam 编译**: `ninja: no work to do.` 然后 `Loading extension module fused_adam... Time to load fused_adam op: 2.185... seconds` (可能已预编译或快速构建)。
*   **DeepSpeed 版本**: `0.9.5`
*   **模型 Dropout 覆盖**:
    *   `Setting model_config.dropout to 0.0`
    *   `Setting model_config.attention_dropout to 0.0`
    *   `Setting model_config.activation_dropout to 0.0`
*   **模型创建时间**: `>Creating model from_config took 38.67... seconds`

### 2. 数据加载与预处理
*   **数据集来源**: `Dahoas/rm-static` (来自 Hugging Face datasets, 已缓存)
*   **`create_prompt_dataset` 的训练阶段**: `2` (对应 Reward Model 训练数据)
*   **训练数据量**: `Creating dataset Dahoas_rm_static for train_phase=2 size=30502` -> **30502 个样本**
*   **评估数据量**: `Creating dataset Dahoas_rm_static for train_phase=2 size=2041` -> **2041 个样本**
*   **Tokenizer 并行警告**: Hugging Face tokenizer 在 fork 环境下的常规警告。

### 3. 训练运行
*   **训练轮次数 (Epochs)**: `1`
*   **每轮总微批次数**: `3813`

## III. 损失与指标变化 (Reward Model 指标)

### 1. 初始评估 (Epoch 0/1 - 训练开始前)
*   **时间戳 (大约)**: `[2025-05-02 11:07:28,xxx]`
*   **Chosen 平均得分 (`chosen_last_scores`)**: `0.8811`
*   **Rejected 平均得分 (`rejected_last_scores`)**: `0.9076`
*   **准确率 (`acc`)**: `0.4750`
    *   *观察: 初始时，rejected 得分略高于 chosen 得分，准确率低于0.5，这符合未训练 RM 的预期。*

### 2. 训练过程中 (Epoch 1/1)
*   **梯度溢出**:
    *   `[2025-05-02 11:07:46,344] [INFO] [fused_optimizer.py:362:_update_scale] Grad overflow on iteration 0`
    *   `[2025-05-02 11:07:46,345] [INFO] [fused_optimizer.py:363:_update_scale] Reducing dynamic loss scale from 65536 to 32768.0`
    *   `[2025-05-02 11:07:46,345] [INFO] [logging.py:96:log_dist] [Rank 0] Overflow detected. Skipping step. Attempted loss scale: 65536, reducing to 32768.0`
*   **Epoch 1 平均损失**: `0.6695552478773276` (在 epoch 结束时报告)

### 3. 最终评估 (Epoch 1/1 - 训练后)
*   **时间戳 (大约)**: 在 "Epoch 1/1 with loss..." 消息之后，"saving model..." 之前
*   **Chosen 平均得分 (`chosen_last_scores`)**: `5.7730`
*   **Rejected 平均得分 (`rejected_last_scores`)**: `5.5493`
*   **准确率 (`acc`)**: `0.6087`
    *   *观察: 训练后，chosen 得分现在高于 rejected 得分，准确率提升至0.5以上，表明模型已学会区分偏好的回复。*

## IV. 变化总结与训练时长

*   Reward Model 训练了 **1 个 epoch**。
*   模型成功学会了区分 "chosen" 和 "rejected" 回复，体现在：
    *   **Chosen 得分显著增加**并高于 rejected 得分。
    *   **准确率**从 `0.4750` 提升至 `0.6087`。
*   该 epoch 的**平均训练损失**约为 `0.670`。
*   在训练刚开始时（迭代0次）发生了一次**梯度溢出**，动态损失缩放器相应地进行了调整。
*   从命令执行 (`[2025-05-02 11:04:43,603]`) 到 "saving model ..." 消息 (在 `[2025-05-02 11:07:46,xxx]` 之后) 的总时间约为 **3 分钟**。单个 epoch 的实际训练循环是此持续时间的一部分。
</details>


<details>
<summary>Step3 PPO-RLHF</summary>

</details>


