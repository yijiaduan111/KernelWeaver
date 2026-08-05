# Selected-15 Ablation Results


## 2. 实验设置

| 项目 | 设置 |
|---|---|
| 任务集合 | L1: P1, P10, P25, P30, P40, P50, P72, P97; L2: P16, P21, P49, P57, P69, P75, P92 |
| 尝试次数 | K=10，统计前10个真实候选；排除 `root` / `phase2_root` |
| 消融A | `flat_search`: `search_profile=main_flat`，有多模型 strategy portfolio，无 phase2 re-root |
| 消融B | `no_delib`: `search_profile=main`，无 strategy portfolio，保留 phase2 re-root |


## 3. 宏观指标

| Metric | 完整框架(同15题) | 消融A: flat_search | 消融B: no_delib |
|---|---:|---:|---:|
| Compile Rate | 100.00% | 100.00% | 100.00% |
| Correct Rate | 100.00% | 100.00% | 100.00% |
| Success@10 | 100.00% | 100.00% | 100.00% |
| Fast1@10 | 100.00% | 86.67% | 86.67% |
| Geomean Speedup | 1.971 | 1.741 | 1.606 |
| Median Speedup | 2.175 | 1.626 | 1.388 |
| Best Speedup | 5.558 | 6.066 | 5.333 |
| Mean Speedup | 2.179 | 1.975 | 1.850 |
| First Correct Median | 2 | 2 | 1 |
| Refine Gain Median | 1.184 | 1.028 | 1.165 |

补充计数：完整框架 Solved/Fast = 15/15、15/15；`flat_search` = 15/15、13/15；`no_delib` = 15/15、13/15。

### 四配置消融表



| Configuration | Agents | Pyramid | Success | Fast1 | GM |
|---|---|---|---:|---:|---:|
| Standalone sampling | No | No | 66.67% | 46.67% | 1.184 |
| Single-agent Pyramid | No | Yes | 100.00% | 86.67% | 1.606 |
| Multi-agent flat | Yes | No | 100.00% | 86.67% | 1.741 |
| PyramidKernel | Yes | Yes | 100.00% | 100.00% | 1.971 |

## 4. 逐题结果

| Task | Name | 完整框架 Best | flat_search Best | no_delib Best | flat C/OK | no_delib C/OK | 备注 |
|---|---|---:|---:|---:|---:|---:|---|
| L1P1 | square_matrix_multiplication | 1.421 | 0.980 | 1.013 | 3/3 | 10/10 | 最高=完整框架; flat<1 |
| L1P10 | 3d_tensor_matrix_multiplication | 1.548 | 1.533 | 1.000 | 5/5 | 10/10 | 最高=完整框架 |
| L1P25 | swish | 2.345 | 2.416 | 2.430 | 9/8 | 10/10 | 最高=no_delib |
| L1P30 | softsign | 3.232 | 2.369 | 2.381 | 10/9 | 10/10 | 最高=完整框架 |
| L1P40 | layernorm | 5.558 | 6.066 | 5.333 | 10/6 | 10/8 | 最高=flat |
| L1P50 | conv_standard_2d_square_input_square_kernel | 1.169 | 0.993 | 1.202 | 6/2 | 10/7 | 最高=no_delib; flat<1 |
| L1P72 | conv_transposed_3d_asymmetric_input_asymmetric_kernel_strided_padded_grouped | 2.488 | 1.626 | 0.953 | 5/4 | 9/9 | 最高=完整框架; no_delib<1 |
| L1P97 | scaleddotproductattention | 2.175 | 2.426 | 1.163 | 3/3 | 9/9 | 最高=flat |
| L2P16 | convtranspose2d_mish_add_hardtanh_scaling | 1.692 | 1.683 | 1.683 | 10/9 | 10/10 | 最高=完整框架 |
| L2P21 | conv2d_add_scale_sigmoid_groupnorm | 1.763 | 1.692 | 1.704 | 8/5 | 7/4 | 最高=完整框架 |
| L2P49 | convtranspose3d_softmax_sigmoid | 1.008 | 1.135 | 0.992 | 9/9 | 9/8 | 最高=flat; no_delib<1 |
| L2P57 | conv2d_relu_hardswish | 2.670 | 2.720 | 1.920 | 10/10 | 9/8 | 最高=flat |
| L2P69 | conv2d_hardswish_relu | 2.190 | 1.342 | 1.388 | 9/9 | 10/10 | 最高=完整框架 |
| L2P75 | gemm_groupnorm_min_biasadd | 1.090 | 1.115 | 1.109 | 10/9 | 9/9 | 最高=flat |
| L2P92 | conv2d_groupnorm_tanh_hardswish_residualadd_logsumexp | 2.336 | 1.524 | 3.482 | 9/8 | 10/10 | 最高=no_delib |




## 5. 源路径

| Group | Run Dir |
|---|---|
| flat_search | `/data/dyj/KernelWeaver/runs/ablation_selected15_two_stage_20260728_013223/flat_search` |
| flat_search retry | `/data/dyj/KernelWeaver/runs/ablation_selected15_two_stage_20260728_013223/retry/flat_search_L1P1` |
| no_delib | `/data/dyj/KernelWeaver/runs/ablation_selected15_two_stage_20260728_013223/no_delib` |
| no_delib retry | `/data/dyj/KernelWeaver/runs/ablation_selected15_two_stage_20260728_013223/retry/no_delib_L1P40` |
