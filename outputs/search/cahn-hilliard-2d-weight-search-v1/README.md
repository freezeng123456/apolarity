# 二维 Cahn--Hilliard 权重搜索交付包

本目录保存 cahn_hilliard_2d_o4/o6 的完整二维 loss-weight 搜索结果。

## 协议

- 权重网格：lambda_ic, lambda_bc ∈ {1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3}，每个任务 49 个 candidate。
- 方法：WAR (complex64, Waring/Taylor jet) 与 real_sinh_autodiff (float32, direct autodiff)。
- 共同网络：4 层、hidden=128、sinh、common Xavier、仿射输入 (x,y,t)，无频率匹配/三角输入。
- 每个 method cell：60 s；采样：n_int=512, n_ic=256, n_bc=512, n_eval=4096, history_eval_n=1024；单 GPU 串行。
- CH 正式 1200 s 任务未启动；本包只包含 60 s 搜参。

## 完整性

- SEARCH_COMPLETE、queue QUEUE_COMPLETE 均存在。
- 两个任务均为 49/49 paired candidates，共 196/196 complete method runs。
- 所有 loss、rel_error、history 有限；无 failed JSON、临时文件或非原子结果。
- 远端 search/SHA256SUMS 已在服务器端通过 sha256sum -c --quiet；本地交付树另有 DELIVERY_SHA256SUMS。
- queue/basic_smoke_conclusion.json 与 queue/search_sized_smoke_conclusion.json 均 passed，raw smoke bundle 未保留。

## 目录

- search/：远端原始搜索树（JSON、逐点日志、rankings、run_matrix、manifest、summary、SHA256SUMS）。
- history/：从原子 JSON 提取的逐 cell history，原始 JSON 未改写。
- analysis/top10_all.csv：四类 ranking 的 Top10。
- analysis/final_metrics_supplement.csv/json：76 个旧日志的补充 final_metrics；数据源是对应原子 JSON，raw log 保持原样。
- analysis/*_weight_sensitivity.csv：按 candidate 的方法误差与四类 rank。
- analysis/method_conflict.csv：方法排名相关性、Top10 重叠与共同阈值统计。
- poly_snapshot/：Poly 正式实验的 marker、manifest、progress、主日志和启动计划。
- queue/：队列状态、主日志、两份 smoke conclusion 和完成 marker。

注意：smoke 只用于 CUDA/有限性/数据管线门禁，不作为论文精度结果。
