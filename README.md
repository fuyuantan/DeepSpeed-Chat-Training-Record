# My DeepSpeed-Chat Training Record
DeepSpeed-Chat: https://github.com/deepspeedai/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat/

<details>
<summary>Step1 SFT</summary>

# Actor Model `facebook/opt-1.3b` 训练日志分析
    
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

# Reward Model `facebook/opt-350m` 训练日志分析

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
    *   **初始学习率 (`lr`)**: `5e-05`
    *   **初始动量 (`mom`)**: `(0.9, 0.95)`
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
*   从命令执行 (`[2025-05-02 11:04:43,603]`) 到 "saving model ..." 消息 (`[2025-05-02 11:24:21,xxx]` ) 的总时间约为 **20 分钟**。单个 epoch 的实际训练循环是此持续时间的一部分。
  
</details>


<details>
<summary>Step3 PPO-RLHF</summary>

# PPO RLHF (第三阶段) 训练日志分析

## I. 参数设置

### 1. 命令行参数与关键参数
*   **Actor 模型 (`actor_model_name_or_path`)**: `/root/DeepSpeedExamples/applications/DeepSpeed-Chat/output/actor-models/1.3b/` (第一阶段 SFT 产出的模型)
*   **Critic 模型 (`critic_model_name_or_path`)**: `/root/DeepSpeedExamples/applications/DeepSpeed-Chat/output/reward-models/350m/` (第二阶段 RM 产出的模型，用作 Critic 和 Reward 打分)
*   **Actor ZeRO 阶段 (`actor_zero_stage`)**: `0`
*   **Critic ZeRO 阶段 (`critic_zero_stage`)**: `0`
*   **起始填充数 (`num_padding_at_beginning`)**: `1`
*   **梯度累积步数 (`gradient_accumulation_steps`)**: `4` (这是用于 DeepSpeed 配置的，PPO 内部可能还有自己的迭代逻辑)
*   **Actor LoRA 维度 (`actor_lora_dim`)**: `128`
*   **Actor 梯度检查点 (`actor_gradient_checkpointing`)**: `True`
*   **Actor Dropout (`actor_dropout`)**: `0.0`
*   **输出目录 (`output_dir`)**: `./output`
*   **TensorBoard**: 未在命令行中显式启用 (但代码中有 `enable_tensorboard` 参数，此处为 False)。
*   **无监督训练**: 未启用 (从 `Unsupervised Loss: 0.0` 和相关参数缺失判断)。
*   **PPO Epochs (`ppo_epochs` 来自代码默认值)**: `1` (每个经验数据批次，PPO 算法迭代训练的次数)
*   **生成批次数 (`generation_batches` 来自代码默认值)**: `1` (收集多少批经验数据后进行一次 PPO 训练)
*   **每设备生成批次大小 (`per_device_generation_batch_size` 来自代码默认值)**: `16`
*   **每设备训练批次大小 (`per_device_training_batch_size` 来自代码默认值)**: `16`

### 2. 分布式训练 (DeepSpeed Launcher)
*   **节点信息 (`world_info`)**: `{'localhost': [0]}` (单 GPU 训练)
*   **主节点地址 (`master_addr`)**: `127.0.0.1`
*   **主节点端口 (`master_port`)**: `29500`
*   **节点数量 (`nnodes`)**: `1`
*   **本地进程数 (`num_local_procs`)**: `1`
*   **分布式世界大小 (`dist_world_size`)**: `1`
*   **可见CUDA设备 (`CUDA_VISIBLE_DEVICES`)**: `0`

### 3. DeepSpeed 配置 (Actor 和 Critic - 两者配置相似)
*   **批处理大小 (每个模型, 来自JSON):**
    *   **每个GPU的训练微批次大小 (`train_micro_batch_size_per_gpu`)**: `8`
    *   **有效训练批次大小 (`train_batch_size`)**: `32`
        *   *(计算: 8 微批次/GPU * 1 GPU * 4 累积步数 = 32)*
*   **ZeRO 优化 (每个模型, 来自JSON):**
    *   **阶段 (`zero_optimization.stage`)**: `0`
    *   参数卸载 (`offload_param.device`): `none`
    *   优化器卸载 (`offload_optimizer.device`): `none`
*   **精度 (每个模型, 来自JSON):**
    *   **FP16 启用 (`fp16.enabled`)**: `True`
    *   损失缩放窗口 (`loss_scale_window`): `100`
    *   初始动态缩放 (`initial_dynamic_scale`): `65536`
*   **优化器与学习率调度器 (Actor 模型, 来自日志):**
    *   **使用的客户端优化器**: `FusedAdam`
    *   **使用的客户端学习率调度器**: `torch.optim.lr_scheduler.LambdaLR`
    *   **初始学习率 (`lr`)**: `[0.0, 0.0, 0.0]` (这通常意味着学习率由 PPO 内部或 `DeepSpeedRLHFEngine` 控制，而非直接使用命令行参数 `actor_learning_rate` 初始化 DeepSpeed 引擎，或者是在 warm-up 阶段)
    *   **初始动量 (`mom`)**: `[(0.9, 0.95), (0.9, 0.95), (0.9, 0.95)]`
*   **优化器与学习率调度器 (Critic 模型, 来自日志 - 类似 Actor):**
    *   **初始学习率 (`lr`)**: `[0.0, 0.0]`
*   **梯度 (每个模型, 来自JSON):**
    *   **梯度裁剪 (`gradient_clipping`)**: `1.0`
*   **TensorBoard (每个模型, 来自JSON):**
    *   `enabled: False` (与命令行参数一致)

## II. 训练细节与执行过程

### 1. 环境与设置
*   **加速器 (`ds_accelerator`)**: `cuda` (自动检测)
*   **主机文件**: 未找到。
*   **Python 环境**: `/root/miniconda3/bin/python`
*   **DeepSpeed 版本**: `0.9.5`
*   **FusedAdam 编译**: `ninja: no work to do.` (已编译或快速构建)
    *   Actor FusedAdam 加载时间: `0.83秒`
    *   Critic FusedAdam 加载时间: `0.001秒` (非常快，可能已加载)

### 2. 模型初始化时长
*   **Actor 模型初始化**: `9.49秒` (包含了LoRA转换)
    *   Actor Dropout 覆盖: `dropout`, `attention_dropout`, `activation_dropout` 设为 `0.0`。
*   **Reference (Ref) 模型初始化**: `3.37秒`
*   **Critic 模型初始化**: `5.74秒`
*   **Reward Model (RM) 初始化**: `5.30秒`
    *   *注意: Critic 和 Reward Model 在此阶段从同一路径加载 (`critic_model_name_or_path`)，但被实例化为 engine 内的不同角色，RM 用于打分，Critic 用于价值估计。它们的 DeepSpeed 配置也可能不同。*

### 3. 数据与训练循环
*   **数据集来源**: `Dahoas/rm-static` (默认路径，脚本内 `create_datasets` 会根据 `train_phase=3` 选择数据)
*   **总迭代次数 (`total_iters`)**: `774` (计算得出，用于学习率调度等)
*   **每 Epoch 总生成批次数**: `1548`
*   **训练 Epoch 数 (`num_train_epochs` 来自代码默认值)**: `1`
*   **每个经验批次的 PPO Epoch 数 (`args.ppo_epochs`)**: `1`

### 4. 性能指标 (代表性的第0步)
*   **端到端延迟**: `14.64秒`
*   **端到端 TFLOPs**: `7.39`
*   **每秒样本数 (Samples/sec)**: `1.09`
*   **生成延迟**: `13.43秒`
    *   每 Token 延迟: `52.46 毫秒`
    *   生成 TFLOPs: `1.63`
    *   答案序列长度: `256` (与 `max_answer_seq_len` 默认值一致)
*   **训练延迟 (PPO 更新)**: `1.21秒`
    *   训练 TFLOPs: `71.56`
*   **模型参数量**:
    *   Actor 模型: `1.429 B` (1.3B 基础模型 + LoRA)
    *   Critic 模型: `0.331 B` (350m 基础模型)

## III. 损失与奖励变化

### 1. 每步指标 (PPO Epoch 1)

| 步骤 | Actor 损失          | Critic 损失         | 平均奖励 (当前批次) | EMA 奖励得分 (全局) |
| :--- | :------------------ | :------------------ | :------------------------- | :------------------------ |
| 0    | `0.05159`           | `0.05991`           | `6.078125`                 | `0.0`                     |
| 1    | `0.03652`           | `0.06056`           | `5.8984375`                | `0.0`                     |
| ...  | ...                 | ...                 | ...                        | ...                       |
| 1547 | `0.00510`           | `0.00043`           | `6.1875`                   | `12.7563`                 |

*   **无监督损失 (Unsupervised Loss)**: `0.0` (始终为0，因为未启用无监督训练)
*   **梯度溢出 (Actor)**:
    *   `[2025-05-02 12:26:02,431] [INFO] [fused_optimizer.py:362:_update_scale] Grad overflow on iteration 0` (针对 Actor 的优化器)
    *   `[2025-05-02 12:26:02,432] [INFO] [fused_optimizer.py:363:_update_scale] Reducing dynamic loss scale from 65536 to 32768.0`
    *   `[2025-05-02 12:26:02,432] [INFO] [logging.py:96:log_dist] [Rank 0] Overflow detected. Skipping step. Attempted loss scale: 65536, reducing to 32768.0`
    *   *注意: 日志中只明确显示了一次 Actor 优化器的梯度溢出，但 `trainer.get_overflow()` 会同时检查 Actor 和 Critic 的溢出情况。*

### 2. 趋势观察
*   **Actor 损失**: 总体呈下降趋势，从初始的 `~0.05` 降低到 `~0.005`。
*   **Critic 损失**: 总体也呈下降趋势，从初始的 `~0.06` 降低到 `~0.0004`。
*   **平均奖励 (每批次)**: 围绕 `6.0` 附近波动，没有非常明显的单向趋势，这在 PPO 训练中是正常的，因为 Actor 在探索和利用之间平衡。
*   **EMA 奖励得分 (指数移动平均)**: 持续上升，从 `0.0` 增加到 `12.7563`，表明 Actor 模型生成的序列平均获得的奖励在稳步提高。这是一个更平滑和更能代表整体学习趋势的指标。注：移动平均 (Exponential Moving Average, EMA)


## IV. 总结与训练时长

*   PPO RLHF 训练成功运行了 **1 个 epoch**，包含 `1548` 个 "生成批次"。
*   **Actor 和 Critic 的损失均显著下降**，表明模型在学习。
*   **EMA 奖励得分显著提升**，表明 Actor 模型生成更高质量（根据 RM 判断）回复的能力在增强。
*   训练开始时 (第0步) Actor 优化器遇到了梯度溢出，动态损失缩放器进行了调整。
*   总训练时长 (从脚本启动到 `saving model ...`):
    *   开始时间: `[2025-05-02 12:24:56,609]` (cmd 执行)
    *   结束时间 (日志中最后一条消息): `[2025-05-02 18:35:55,615]` (进程退出)
    *   大约持续了 **6 小时 11 分钟**。

</details>


