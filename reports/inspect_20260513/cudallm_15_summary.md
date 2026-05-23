# Run Summary

- Run dir: `/data/dyj/KernelWeaver/runs/main_l1_15_cuda_cudallm_delib_main_20260513_110447`
- Tasks with run.json: `6`
- Summary rows: `15`
- task_count: `15`
- success_count: `6`
- compile_rate: `0.5333333333333333`
- correct_rate: `0.36666666666666664`
- best_speedup: `2.1047120418848166`
- median_speedup: `1.0026018423401426`
- improved_over_reference_rate: `0.26666666666666666`
- paper_metrics: `{"Fast1": 0.4, "Speed": 0.5123897972437862, "Success": 0.4}`

| Task | Speedup | Best | Attempts | Compile OK | Correct | Strategies | Op Type |
|---|---:|---|---:|---:|---:|---:|---|
| `L1_P10_TensorMatmul3D_l1_p10` | `1.001002004008016` | `root` | `10` | `8` | `4` | `10` | `matmul` |
| `L1_P1_SquareMatmul_l1_p1` | `1.0042016806722691` | `n8` | `10` | `5` | `4` | `10` | `matmul` |
| `L1_P20_LeakyReLU_l1_p20` | `1.0` | `root` | `10` | `4` | `4` | `8` | `elementwise` |
| `L1_P25_Swish_l1_p25` | `2.1047120418848166` | `n3` | `10` | `8` | `8` | `10` | `elementwise` |
| `L1_P40_LayerNorm_l1_p40` | `1.0` | `root` | `10` | `5` | `4` | `10` | `normalization` |
| `L1_P42_MaxPool2d_l1_p42` | `1.5759312320916905` | `n2` | `10` | `8` | `4` | `10` | `reduction` |
