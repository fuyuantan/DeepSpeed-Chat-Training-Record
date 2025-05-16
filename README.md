# DeepSpeed-Chat-Training-Record
DeepSpeed-Chat from https://github.com/deepspeedai/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat/

<details>
<summary>Step1 SFT</summary>

I. 参数设置:<br>
1.模型与训练策略:<br>
  基础模型 (model_name_or_path): facebook/opt-1.3b<br>
  LoRA 维度 (lora_dim): 128<br>
  梯度累积步数 (gradient_accumulation_steps): 16<br>
2.分布式训练 (DeepSpeed Launcher):<br>
  执行命令 (cmd): 包含了完整的启动命令，指明使用 deepspeed.launcher.launch<br>
  节点信息 (world_info): {'localhost': [0]} (表示在本地机器的 GPU 0 上训练)<br>
  主节点地址 (master_addr): 127.0.0.1<br>
  主节点端口 (master_port): 29500<br>
  节点数量 (nnodes): 1<br>
  本地进程数 (num_local_procs): 1 (即使用1个GPU)<br>
  总进程数/世界大小 (dist_world_size): 1<br>
  可见CUDA设备 (CUDA_VISIBLE_DEVICES): 0<br>
3.DeepSpeed 配置 (从日志中的 DeepSpeedEngine configuration 和 json 部分提取):<br>
    批处理大小:<br>
        训练总批次大小 (train_batch_size): 128<br>
        每个GPU的微批次大小 (train_micro_batch_size_per_gpu): 8(验证: 微批次 8 * 梯度累积 16 = 128，与总批次大小相符)<br>
    ZeRO 优化:<br>
        ZeRO 阶段 (zero_stage / zero_optimization.stage): 0 (表示优化器状态和梯度不进行分片)<br>
        参数卸载 (offload_param.device): none (参数不卸载到CPU/NVMe)<br>
        优化器状态卸载 (offload_optimizer.device): none (优化器状态不卸载)
    精度控制:<br>
        FP16 启用 (fp16.enabled): True (启用了混合精度训练)<br>
        FP16 损失缩放窗口 (fp16.loss_scale_window): 100<br>
        初始动态损失缩放值 (initial_dynamic_scale / dynamic_loss_scale_args.init_scale): 65536<br>
    优化器与学习率调度器:<br>
        使用的客户端优化器: FusedAdam (DeepSpeed 提供的融合 Adam 优化器)<br>
        使用的客户端学习率调度器: torch.optim.lr_scheduler.LambdaLR<br>
        初始学习率 (lr): 0.001<br>
        初始动量 (mom): (0.9, 0.95)<br>
    梯度处理:<br>
        梯度裁剪 (gradient_clipping): 1.0<br>
        梯度预缩放 (prescale_gradients): false<br>
    日志与监控:<br>
        打印间隔步数 (steps_per_print): 10<br>
        TensorBoard 启用 (tensorboard.enabled): True<br>
        TensorBoard 输出路径 (tensorboard.output_path): /root/DeepSpeedExamples/applications/DeepSpeed-Chat/output/actor-models/1.3b/ds_tensorboard_logs/<br>
        TensorBoard 任务名称 (tensorboard.job_name): step1_model_tensorboard<br>
    输出目录:<br>
        模型输出目录 (output_dir): /root/DeepSpeedExamples/applications/DeepSpeed-Chat/output/actor-models/1.3b<br>

II. 训练细节与执行过程:<br>
    1.环境与设置:<br>
        加速器 (ds_accelerator): cuda (自动检测)<br>
        Python 环境: /root/miniconda3/bin/python<br>
        主机文件 (hostfile): 未找到，仅使用本地资源进行训练。<br>
        TorchVision Beta 版本警告: 关于 torchvision.datapoints 和 torchvision.transforms.v2 仍处于 Beta 阶段的常规警告。<br>
        CUDA 版本: 安装版本 11.8，PyTorch 编译版本 11.7。日志认为此组合兼容。<br>
        DeepSpeed NCCL 后端: 日志先是提到 NCCL backend in DeepSpeed not yet implemented，但随后 Initializing TorchBackend in DeepSpeed with backend nccl。这表明 DeepSpeed 正在利用 PyTorch 的 NCCL 后端进行通信。<br>
        FusedAdam 算子编译: fused_adam 优化器算子进行了即时编译，耗时 50.531 秒。<br>
        DeepSpeed 版本: 0.9.5<br>
    2.训练运行:<br>
        训练轮次 (Epochs): 1 (日志显示 Epoch 0/1，然后是 Epoch 1/1)<br>
        每轮的总微批次数 (Total Micro Batches): 1907<br>
        报告的模型参数量: 1.429 B (这应包括基础模型 1.3B 加上 LoRA 引入的参数)<br>
        序列长度 (Sequence Length): 512<br>
        
III. 损失 (Loss) 与困惑度 (Perplexity) 变化:<br>
    1.初始评估 (第0轮结束/第1轮开始前):<br>
        时间戳: [2025-05-02 10:52:48,xxx] (大约)<br>
        困惑度 (ppl): 8.376730918884277<br>
        损失 (loss): 2.125457763671875<br>
    2.训练过程中 (第1轮):<br>
        日志大约每 10 步打印一次训练指标 (根据 steps_per_print 设置)。<br>
        性能指标 (从日志中提取的代表性数值，存在波动):<br>
        每个微批次的延迟 (Latency): 约 0.33 秒 (例如 0.34s, 0.33s, 也有较低的 0.18s 和较高的 0.39s)<br>
        TFLOPs (每秒万亿次浮点运算): 约 70-71 (例如 67.75, 70.49, 71.32, 也有较低的 59.26 和一次较高的 127.52)<br>
        Samples/sec (每秒处理样本数): 约 24 (例如 23.31, 24.26, 24.54, 也有较低的 20.39 和一次较高的 43.88)<br>
        Time/seq (处理单个序列时间): 约 0.04 秒<br>
注意: 性能指标的显著波动 (如0.18s延迟, 127.52 TFLOPs) 可能是由于最后一个批次较小、系统波动或特定步骤的特殊操作导致。<br>
    3.最终评估 (第1轮结束):<br>
        时间戳: [2025-05-02 11:04:35,xxx] (大约，在保存模型之前)<br>
        困惑度 (ppl): 5.937998294830322<br>
        损失 (loss): 1.7813720703125<br>

**总结与变化**:<br>
训练总共进行了 1 个 epoch。<br>
损失 (loss) 从训练开始时的 2.125 (初始评估) 降低到 epoch 结束时的 1.781。<br>
相应地，困惑度 (perplexity, ppl) 从 8.377 改善至 5.938。<br>
主要训练循环和最终评估过程大约耗时: 11:04:35 - 10:52:48 = 11 分 47 秒 (不包括初始的环境设置和 FusedAdam 编译时间)。<br>


</details>
