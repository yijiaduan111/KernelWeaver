# Run Summary

- Run dir: `/data/dyj/KernelWeaver/runs/main_l1_15_cuda_claude_delib_main_20260513_124238`
- Tasks with run.json: `15`
- Summary rows: `15`
- task_count: `15`
- success_count: `3`
- compile_rate: `1.0`
- correct_rate: `0.13333333333333333`
- best_speedup: `1.0304878048780488`
- median_speedup: `1.0`
- improved_over_reference_rate: `0.06666666666666667`
- paper_metrics: `{"Fast1": 0.13333333333333333, "Speed": 0.20119570024152125, "Success": 0.2}`

| Task | Speedup | Best | Attempts | Compile OK | Correct | Strategies | Op Type |
|---|---:|---|---:|---:|---:|---:|---|
| `L1_P10_TensorMatmul3D_l1_p10` | `1.0` | `n2` | `10` | `11` | `7` | `10` | `matmul` |
| `L1_P1_SquareMatmul_l1_p1` | `0.9874476987447697` | `root` | `10` | `11` | `7` | `10` | `matmul` |
| `L1_P20_LeakyReLU_l1_p20` | `1.0304878048780488` | `n2` | `10` | `11` | `9` | `10` | `elementwise` |
| `L1_P25_Swish_l1_p25` | `None` | `root` | `10` | `11` | `0` | `9` | `elementwise` |
| `L1_P33_BatchNorm_l1_p33` | `None` | `root` | `10` | `11` | `0` | `10` | `normalization` |
| `L1_P40_LayerNorm_l1_p40` | `None` | `root` | `10` | `11` | `0` | `10` | `normalization` |
| `L1_P42_MaxPool2d_l1_p42` | `None` | `root` | `10` | `11` | `0` | `10` | `reduction` |
| `L1_P45_AvgPool2d_l1_p45` | `None` | `root` | `10` | `11` | `0` | `7` | `pooling` |
| `L1_P47_SumReduction_l1_p47` | `None` | `root` | `10` | `11` | `0` | `8` | `reduction` |
| `L1_P50_Conv2dStandard_l1_p50` | `None` | `root` | `10` | `11` | `0` | `8` | `convolution` |
| `L1_P61_ConvTranspose3d_l1_p61` | `None` | `root` | `10` | `11` | `0` | `8` | `unknown` |
| `L1_P82_DepthwiseConv2d_l1_p82` | `None` | `root` | `10` | `11` | `0` | `10` | `convolution` |
| `L1_P89_Cumsum_l1_p89` | `None` | `root` | `10` | `11` | `0` | `8` | `reduction` |
| `L1_P95_CrossEntropyLoss_l1_p95` | `None` | `root` | `10` | `11` | `0` | `10` | `loss` |
| `L1_P97_ScaledDotProductAttention_l1_p97` | `None` | `root` | `10` | `11` | `0` | `10` | `attention` |
