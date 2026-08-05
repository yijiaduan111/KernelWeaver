# Direct CUDALLM KernelBench Snapshot



## Summary

- Accepted tasks: `250/250`
- Level 1: `100/100`
- Level 2: `100/100`
- Level 3: `50/50`
- Missing tasks: none
- Note: this CUDALLM baseline uses K=10 candidates per task, so `Success@10` and `Fast1@10` are task-level outcomes over up to 10 generated candidates.

## Metrics by Level

| Metric | Overall | L1 | L2 | L3 |
|---|---:|---:|---:|---:|
| Compile Rate | 0.9040 | 0.9400 | 0.9100 | 0.8200 |
| Correct Rate | 0.8960 | 0.8600 | 0.9700 | 0.8200 |
| Success@10 | 0.8960 | 0.8600 | 0.9700 | 0.8200 |
| Fast1@10 | 0.6200 | 0.6800 | 0.7000 | 0.3400 |
| Geomean Speedup | 1.1884x | 1.1505x | 1.3135x | 1.0037x |
| Median Speedup | 1.0000x | 1.0000x | 1.0238x | 0.9964x |
| Best Speedup | 944.4444x | 9.1705x | 944.4444x | 2.5498x |
| Mean Speedup | 5.6893x | 1.2908x | 11.5544x | 1.0391x |
| First Correct Median | 3.0000 | 3.0000 | 3.0000 | 3.0000 |
| Refine Gain Median | 1.0017x | 1.0000x | 1.0002x | 1.0040x |

## Per-Task Best Correct Speedup

| Level | Problem | Task Name | Best Correct Speedup | Best Node |
|---:|---:|---|---:|---|
| L1 | P1 | kernelbench_l1_1_1_square_matrix_multiplication | 1.0042x | root |
| L1 | P2 | kernelbench_l1_2_2_standard_matrix_multiplication | 1.0000x | n2 |
| L1 | P3 | kernelbench_l1_3_3_batched_matrix_multiplication | 1.7945x | phase2_n3 |
| L1 | P4 | kernelbench_l1_4_4_matrix_vector_multiplication | 1.0010x | root |
| L1 | P5 | kernelbench_l1_5_5_matrix_scalar_multiplication | N/A | N/A |
| L1 | P6 | kernelbench_l1_6_6_matmul_with_large_k_dimension | 1.0000x | root |
| L1 | P7 | kernelbench_l1_7_7_matmul_with_small_k_dimension | N/A | N/A |
| L1 | P8 | kernelbench_l1_8_8_matmul_with_irregular_shapes | 1.4493x | n5 |
| L1 | P9 | kernelbench_l1_9_9_tall_skinny_matrix_multiplication | N/A | N/A |
| L1 | P10 | kernelbench_l1_10_10_3d_tensor_matrix_multiplication | 1.0000x | root |
| L1 | P11 | kernelbench_l1_11_11_4d_tensor_matrix_multiplication | 1.0060x | n3 |
| L1 | P12 | kernelbench_l1_12_12_matmul_with_diagonal_matrices | 1.0181x | phase2_n2 |
| L1 | P13 | kernelbench_l1_13_13_matmul_for_symmetric_matrices | 0.9123x | n4 |
| L1 | P14 | kernelbench_l1_14_14_matmul_for_upper_triangular_matrices | 1.0000x | root |
| L1 | P15 | kernelbench_l1_15_15_matmul_for_lower_triangular_matrices | 0.9551x | root |
| L1 | P16 | kernelbench_l1_16_16_matmul_with_transposed_a | 1.0047x | root |
| L1 | P17 | kernelbench_l1_17_17_matmul_with_transposed_b | 0.9969x | root |
| L1 | P18 | kernelbench_l1_18_18_matmul_with_transposed_both | 1.0000x | root |
| L1 | P19 | kernelbench_l1_19_19_relu | 0.9941x | root |
| L1 | P20 | kernelbench_l1_20_20_leakyrelu | 1.0000x | root |
| L1 | P21 | kernelbench_l1_21_21_sigmoid | 1.0000x | root |
| L1 | P22 | kernelbench_l1_22_22_tanh | 1.0000x | root |
| L1 | P23 | kernelbench_l1_23_23_softmax | 1.0000x | root |
| L1 | P24 | kernelbench_l1_24_24_logsoftmax | 1.0000x | root |
| L1 | P25 | kernelbench_l1_25_25_swish | 2.1211x | phase2_n5 |
| L1 | P26 | kernelbench_l1_26_26_gelu | 1.0303x | n4 |
| L1 | P27 | kernelbench_l1_27_27_selu | 1.0000x | root |
| L1 | P28 | kernelbench_l1_28_28_hardsigmoid | 1.0000x | root |
| L1 | P29 | kernelbench_l1_29_29_softplus | 1.0000x | root |
| L1 | P30 | kernelbench_l1_30_30_softsign | 2.8872x | phase2_n3 |
| L1 | P31 | kernelbench_l1_31_31_elu | 0.9884x | phase2_n2 |
| L1 | P32 | kernelbench_l1_32_32_hardtanh | 1.0366x | n2 |
| L1 | P33 | kernelbench_l1_33_33_batchnorm | 1.0000x | n9 |
| L1 | P34 | kernelbench_l1_34_34_instancenorm | 1.7987x | n5 |
| L1 | P35 | kernelbench_l1_35_35_groupnorm | 0.3886x | n9 |
| L1 | P36 | kernelbench_l1_36_36_rmsnorm | 2.0082x | phase2_n2 |
| L1 | P37 | kernelbench_l1_37_37_frobeniusnorm | 1.2630x | n4 |
| L1 | P38 | kernelbench_l1_38_38_l1norm | N/A | N/A |
| L1 | P39 | kernelbench_l1_39_39_l2norm | N/A | N/A |
| L1 | P40 | kernelbench_l1_40_40_layernorm | 0.9820x | n10 |
| L1 | P41 | kernelbench_l1_41_41_max_pooling_1d | 1.2620x | n8 |
| L1 | P42 | kernelbench_l1_42_42_max_pooling_2d | 1.0776x | phase2_n2 |
| L1 | P43 | kernelbench_l1_43_43_max_pooling_3d | 1.0000x | phase2_n4 |
| L1 | P44 | kernelbench_l1_44_44_average_pooling_1d | 1.0565x | root |
| L1 | P45 | kernelbench_l1_45_45_average_pooling_2d | 1.0000x | root |
| L1 | P46 | kernelbench_l1_46_46_average_pooling_3d | 1.1453x | root |
| L1 | P47 | kernelbench_l1_47_47_sum_reduction_over_a_dimension | 1.0000x | root |
| L1 | P48 | kernelbench_l1_48_48_mean_reduction_over_a_dimension | 1.0000x | root |
| L1 | P49 | kernelbench_l1_49_49_max_reduction_over_a_dimension | 1.1221x | root |
| L1 | P50 | kernelbench_l1_50_50_conv_standard_2d_square_input_square_kernel | 1.3271x | n2 |
| L1 | P51 | kernelbench_l1_51_51_argmax_over_a_dimension | 1.2400x | phase2_n5 |
| L1 | P52 | kernelbench_l1_52_52_argmin_over_a_dimension | 1.0000x | n3 |
| L1 | P53 | kernelbench_l1_53_53_min_reduction_over_a_dimension | 1.0000x | root |
| L1 | P54 | kernelbench_l1_54_54_conv_standard_3d_square_input_square_kernel | 1.5276x | root |
| L1 | P55 | kernelbench_l1_55_55_conv_standard_2d_asymmetric_input_square_kernel | 1.2027x | root |
| L1 | P56 | kernelbench_l1_56_56_conv_standard_2d_asymmetric_input_asymmetric_kernel | 1.0802x | phase2_n3 |
| L1 | P57 | kernelbench_l1_57_57_conv_transposed_2d_square_input_square_kernel | 0.9766x | n4 |
| L1 | P58 | kernelbench_l1_58_58_conv_transposed_3d_asymmetric_input_asymmetric_kernel | 1.0000x | root |
| L1 | P59 | kernelbench_l1_59_59_conv_standard_3d_asymmetric_input_square_kernel | 1.0016x | root |
| L1 | P60 | kernelbench_l1_60_60_conv_standard_3d_square_input_asymmetric_kernel | 0.9973x | phase2_n2 |
| L1 | P61 | kernelbench_l1_61_61_conv_transposed_3d_square_input_square_kernel | 1.0000x | root |
| L1 | P62 | kernelbench_l1_62_62_conv_standard_2d_square_input_asymmetric_kernel | 1.8389x | n8 |
| L1 | P63 | kernelbench_l1_63_63_conv_standard_2d_square_input_square_kernel | N/A | N/A |
| L1 | P64 | kernelbench_l1_64_64_conv_transposed_1d | 1.1165x | n7 |
| L1 | P65 | kernelbench_l1_65_65_conv_transposed_2d_square_input_asymmetric_kernel | 1.0339x | n5 |
| L1 | P66 | kernelbench_l1_66_66_conv_standard_3d_asymmetric_input_asymmetric_kernel | 0.9973x | root |
| L1 | P67 | kernelbench_l1_67_67_conv_standard_1d | 1.0121x | n8 |
| L1 | P68 | kernelbench_l1_68_68_conv_transposed_3d_square_input_asymmetric_kernel | 1.0000x | root |
| L1 | P69 | kernelbench_l1_69_69_conv_transposed_2d_asymmetric_input_asymmetric_kernel | 0.9923x | n5 |
| L1 | P70 | kernelbench_l1_70_70_conv_transposed_3d_asymmetric_input_square_kernel | N/A | N/A |
| L1 | P71 | kernelbench_l1_71_71_conv_transposed_2d_asymmetric_input_square_kernel | 1.0000x | n2 |
| L1 | P72 | kernelbench_l1_72_72_conv_transposed_3d_asymmetric_input_asymmetric_kernel_strided_padded_grouped | 0.9818x | phase2_n2 |
| L1 | P73 | kernelbench_l1_73_73_conv_transposed_3d_asymmetric_input_square_kernel_strided_padded_grouped | 1.0569x | n10 |
| L1 | P74 | kernelbench_l1_74_74_conv_transposed_1d_dilated | 0.9963x | n5 |
| L1 | P75 | kernelbench_l1_75_75_conv_transposed_2d_asymmetric_input_asymmetric_kernel_strided_grouped_padded_dilated | 1.0000x | root |
| L1 | P76 | kernelbench_l1_76_76_conv_standard_1d_dilated_strided | N/A | N/A |
| L1 | P77 | kernelbench_l1_77_77_conv_transposed_3d_square_input_square_kernel_padded_dilated_strided | 1.8448x | n9 |
| L1 | P78 | kernelbench_l1_78_78_conv_transposed_2d_asymmetric_input_asymmetric_kernel_padded | 0.9972x | root |
| L1 | P79 | kernelbench_l1_79_79_conv_transposed_1d_asymmetric_input_square_kernel_padded_strided_dilated | 1.0000x | root |
| L1 | P80 | kernelbench_l1_80_80_conv_standard_2d_square_input_asymmetric_kernel_dilated_padded | 1.0000x | root |
| L1 | P81 | kernelbench_l1_81_81_conv_transposed_2d_asymmetric_input_square_kernel_dilated_padded_strided | 1.1617x | phase2_n3 |
| L1 | P82 | kernelbench_l1_82_82_conv_depthwise_2d_square_input_square_kernel | 1.6453x | root |
| L1 | P83 | kernelbench_l1_83_83_conv_depthwise_2d_square_input_asymmetric_kernel | 1.0000x | root |
| L1 | P84 | kernelbench_l1_84_84_conv_depthwise_2d_asymmetric_input_square_kernel | 0.9926x | phase2_n3 |
| L1 | P85 | kernelbench_l1_85_85_conv_depthwise_2d_asymmetric_input_asymmetric_kernel | 0.9111x | root |
| L1 | P86 | kernelbench_l1_86_86_conv_depthwise_separable_2d | 0.9793x | n8 |
| L1 | P87 | kernelbench_l1_87_87_conv_pointwise_2d | N/A | N/A |
| L1 | P88 | kernelbench_l1_88_88_mingptnewgelu | 9.1705x | n2 |
| L1 | P89 | kernelbench_l1_89_89_cumsum | 1.0000x | root |
| L1 | P90 | kernelbench_l1_90_90_cumprod | 0.9851x | root |
| L1 | P91 | kernelbench_l1_91_91_cumsum_reverse | N/A | N/A |
| L1 | P92 | kernelbench_l1_92_92_cumsum_exclusive | N/A | N/A |
| L1 | P93 | kernelbench_l1_93_93_masked_cumsum | N/A | N/A |
| L1 | P94 | kernelbench_l1_94_94_mseloss | 2.9439x | n3 |
| L1 | P95 | kernelbench_l1_95_95_crossentropyloss | N/A | N/A |
| L1 | P96 | kernelbench_l1_96_96_huberloss | N/A | N/A |
| L1 | P97 | kernelbench_l1_97_97_scaleddotproductattention | 1.0145x | root |
| L1 | P98 | kernelbench_l1_98_98_kldivloss | 5.5985x | n3 |
| L1 | P99 | kernelbench_l1_99_99_tripletmarginloss | 1.0127x | n10 |
| L1 | P100 | kernelbench_l1_100_100_hingeloss | 1.0726x | root |
| L2 | P1 | kernelbench_l2_1_1_conv2d_relu_biasadd | 1.3565x | n5 |
| L2 | P2 | kernelbench_l2_2_2_convtranspose2d_biasadd_clamp_scaling_clamp_divide | 1.5761x | n5 |
| L2 | P3 | kernelbench_l2_3_3_convtranspose3d_sum_layernorm_avgpool_gelu | 0.9897x | root |
| L2 | P4 | kernelbench_l2_4_4_conv2d_mish_mish | 1.2268x | phase2_n3 |
| L2 | P5 | kernelbench_l2_5_5_convtranspose2d_subtract_tanh | 0.9579x | root |
| L2 | P6 | kernelbench_l2_6_6_conv3d_softmax_maxpool_maxpool | 1.0000x | root |
| L2 | P7 | kernelbench_l2_7_7_conv3d_relu_leakyrelu_gelu_sigmoid_biasadd | 1.9937x | n2 |
| L2 | P8 | kernelbench_l2_8_8_conv3d_divide_max_globalavgpool_biasadd_sum | 0.9501x | phase2_n4 |
| L2 | P9 | kernelbench_l2_9_9_matmul_subtract_multiply_relu | 1.0169x | phase2_n4 |
| L2 | P10 | kernelbench_l2_10_10_convtranspose2d_maxpool_hardtanh_mean_tanh | 0.9880x | phase2_n1 |
| L2 | P11 | kernelbench_l2_11_11_convtranspose2d_batchnorm_tanh_maxpool_groupnorm | 1.0872x | n8 |
| L2 | P12 | kernelbench_l2_12_12_gemm_multiply_leakyrelu | 1.3082x | n5 |
| L2 | P13 | kernelbench_l2_13_13_convtranspose3d_mean_add_softmax_tanh_scaling | 1.0000x | root |
| L2 | P14 | kernelbench_l2_14_14_gemm_divide_sum_scaling | 0.9547x | root |
| L2 | P15 | kernelbench_l2_15_15_convtranspose3d_batchnorm_subtract | 1.2160x | phase2_n4 |
| L2 | P16 | kernelbench_l2_16_16_convtranspose2d_mish_add_hardtanh_scaling | 1.6850x | phase2_n3 |
| L2 | P17 | kernelbench_l2_17_17_conv2d_instancenorm_divide | 1.0000x | n2 |
| L2 | P18 | kernelbench_l2_18_18_matmul_sum_max_avgpool_logsumexp_logsumexp | 49.8387x | phase2_n3 |
| L2 | P19 | kernelbench_l2_19_19_convtranspose2d_gelu_groupnorm | N/A | N/A |
| L2 | P20 | kernelbench_l2_20_20_convtranspose3d_sum_residualadd_multiply_residualadd | 1.7402x | n5 |
| L2 | P21 | kernelbench_l2_21_21_conv2d_add_scale_sigmoid_groupnorm | 0.9882x | root |
| L2 | P22 | kernelbench_l2_22_22_matmul_scale_residualadd_clamp_logsumexp_mish | 1.1258x | n7 |
| L2 | P23 | kernelbench_l2_23_23_conv3d_groupnorm_mean | 0.8447x | root |
| L2 | P24 | kernelbench_l2_24_24_conv3d_min_softmax | 1.0000x | root |
| L2 | P25 | kernelbench_l2_25_25_conv2d_min_tanh_tanh | 1.0075x | n2 |
| L2 | P26 | kernelbench_l2_26_26_convtranspose3d_add_hardswish | 1.4588x | phase2_n3 |
| L2 | P27 | kernelbench_l2_27_27_conv3d_hardswish_groupnorm_mean | 1.0979x | n3 |
| L2 | P28 | kernelbench_l2_28_28_bmm_instancenorm_sum_residualadd_multiply | 1.0903x | phase2_n3 |
| L2 | P29 | kernelbench_l2_29_29_matmul_mish_mish | 1.0720x | root |
| L2 | P30 | kernelbench_l2_30_30_gemm_groupnorm_hardtanh | 1.1558x | root |
| L2 | P31 | kernelbench_l2_31_31_conv2d_min_add_multiply | 1.5000x | phase2_n2 |
| L2 | P32 | kernelbench_l2_32_32_conv2d_scaling_min | 1.2515x | n2 |
| L2 | P33 | kernelbench_l2_33_33_gemm_scale_batchnorm | 1.0000x | n9 |
| L2 | P34 | kernelbench_l2_34_34_convtranspose3d_layernorm_gelu_scaling | 0.9943x | n10 |
| L2 | P35 | kernelbench_l2_35_35_conv2d_subtract_hardswish_maxpool_mish | 1.6649x | n6 |
| L2 | P36 | kernelbench_l2_36_36_convtranspose2d_min_sum_gelu_add | 0.9836x | root |
| L2 | P37 | kernelbench_l2_37_37_matmul_swish_sum_groupnorm | 0.9779x | root |
| L2 | P38 | kernelbench_l2_38_38_convtranspose3d_avgpool_clamp_softmax_multiply | 1.2918x | phase2_n3 |
| L2 | P39 | kernelbench_l2_39_39_gemm_scale_batchnorm | 1.0137x | n5 |
| L2 | P40 | kernelbench_l2_40_40_matmul_scaling_residualadd | 1.7664x | phase2_n5 |
| L2 | P41 | kernelbench_l2_41_41_gemm_batchnorm_gelu_relu | 1.0238x | root |
| L2 | P42 | kernelbench_l2_42_42_convtranspose2d_globalavgpool_biasadd_logsumexp_sum_multiply | 1.0000x | n5 |
| L2 | P43 | kernelbench_l2_43_43_conv3d_max_logsumexp_relu | 0.9649x | n5 |
| L2 | P44 | kernelbench_l2_44_44_convtranspose2d_multiply_globalavgpool_globalavgpool_mean | 1.2577x | n2 |
| L2 | P45 | kernelbench_l2_45_45_gemm_sigmoid_logsumexp | 1.0093x | n10 |
| L2 | P46 | kernelbench_l2_46_46_conv2d_subtract_tanh_subtract_avgpool | 0.9884x | root |
| L2 | P47 | kernelbench_l2_47_47_conv3d_mish_tanh | 1.3847x | phase2_n2 |
| L2 | P48 | kernelbench_l2_48_48_conv3d_scaling_tanh_multiply_sigmoid | 0.9973x | root |
| L2 | P49 | kernelbench_l2_49_49_convtranspose3d_softmax_sigmoid | 1.0210x | phase2_n5 |
| L2 | P50 | kernelbench_l2_50_50_convtranspose3d_scaling_avgpool_biasadd_scaling | 0.9425x | phase2_n5 |
| L2 | P51 | kernelbench_l2_51_51_gemm_subtract_globalavgpool_logsumexp_gelu_residualadd | 1.0288x | phase2_n2 |
| L2 | P52 | kernelbench_l2_52_52_conv2d_activation_batchnorm | 1.3702x | n4 |
| L2 | P53 | kernelbench_l2_53_53_gemm_scaling_hardtanh_gelu | 1.9716x | n6 |
| L2 | P54 | kernelbench_l2_54_54_conv2d_multiply_leakyrelu_gelu | 1.0000x | root |
| L2 | P55 | kernelbench_l2_55_55_matmul_maxpool_sum_scale | 1.0223x | phase2_n4 |
| L2 | P56 | kernelbench_l2_56_56_matmul_sigmoid_sum | 0.9984x | root |
| L2 | P57 | kernelbench_l2_57_57_conv2d_relu_hardswish | 2.6364x | phase2_n2 |
| L2 | P58 | kernelbench_l2_58_58_convtranspose3d_logsumexp_hardswish_subtract_clamp | 1.4646x | root |
| L2 | P59 | kernelbench_l2_59_59_matmul_swish_scaling | 1.0016x | n4 |
| L2 | P60 | kernelbench_l2_60_60_convtranspose3d_swish_groupnorm_hardswish | 0.9970x | root |
| L2 | P61 | kernelbench_l2_61_61_convtranspose3d_relu_groupnorm | 1.1063x | n10 |
| L2 | P62 | kernelbench_l2_62_62_matmul_groupnorm_leakyrelu_sum | 0.9942x | root |
| L2 | P63 | kernelbench_l2_63_63_gemm_relu_divide | 1.0000x | root |
| L2 | P64 | kernelbench_l2_64_64_gemm_logsumexp_leakyrelu_leakyrelu_gelu_gelu | 1.0263x | phase2_n5 |
| L2 | P65 | kernelbench_l2_65_65_conv2d_avgpool_sigmoid_sum | 0.9691x | root |
| L2 | P66 | kernelbench_l2_66_66_matmul_dropout_softmax | N/A | N/A |
| L2 | P67 | kernelbench_l2_67_67_conv2d_gelu_globalavgpool | 0.9929x | root |
| L2 | P68 | kernelbench_l2_68_68_matmul_min_subtract | 0.9936x | root |
| L2 | P69 | kernelbench_l2_69_69_conv2d_hardswish_relu | 1.4006x | n2 |
| L2 | P70 | kernelbench_l2_70_70_gemm_sigmoid_scaling_residualadd | 1.0461x | phase2_n3 |
| L2 | P71 | kernelbench_l2_71_71_conv2d_divide_leakyrelu | 1.4018x | n2 |
| L2 | P72 | kernelbench_l2_72_72_convtranspose3d_batchnorm_avgpool_avgpool | 1.6019x | phase2_n5 |
| L2 | P73 | kernelbench_l2_73_73_conv2d_batchnorm_scaling | 0.9985x | n3 |
| L2 | P74 | kernelbench_l2_74_74_convtranspose3d_leakyrelu_multiply_leakyrelu_max | 0.9777x | root |
| L2 | P75 | kernelbench_l2_75_75_gemm_groupnorm_min_biasadd | 1.1604x | n4 |
| L2 | P76 | kernelbench_l2_76_76_gemm_add_relu | 1.0721x | phase2_n2 |
| L2 | P77 | kernelbench_l2_77_77_convtranspose3d_scale_batchnorm_globalavgpool | 0.9919x | phase2_n2 |
| L2 | P78 | kernelbench_l2_78_78_convtranspose3d_max_max_sum | 0.9972x | root |
| L2 | P79 | kernelbench_l2_79_79_conv3d_multiply_instancenorm_clamp_multiply_max | 1.0000x | root |
| L2 | P80 | kernelbench_l2_80_80_gemm_max_subtract_gelu | 944.4444x | phase2_n3 |
| L2 | P81 | kernelbench_l2_81_81_gemm_swish_divide_clamp_tanh_clamp | 1.0689x | n5 |
| L2 | P82 | kernelbench_l2_82_82_conv2d_tanh_scaling_biasadd_max | 2.2583x | phase2_n2 |
| L2 | P83 | kernelbench_l2_83_83_conv3d_groupnorm_min_clamp_dropout | 15.4731x | phase2_n2 |
| L2 | P84 | kernelbench_l2_84_84_gemm_batchnorm_scaling_softmax | 1.0000x | root |
| L2 | P85 | kernelbench_l2_85_85_conv2d_groupnorm_scale_maxpool_clamp | 0.9856x | root |
| L2 | P86 | kernelbench_l2_86_86_matmul_divide_gelu | 1.0000x | n2 |
| L2 | P87 | kernelbench_l2_87_87_conv2d_subtract_subtract_mish | 1.7926x | n3 |
| L2 | P88 | kernelbench_l2_88_88_gemm_groupnorm_swish_multiply_swish | 1.0409x | phase2_n3 |
| L2 | P89 | kernelbench_l2_89_89_convtranspose3d_maxpool_softmax_subtract_swish_max | 1.7899x | n6 |
| L2 | P90 | kernelbench_l2_90_90_conv3d_leakyrelu_sum_clamp_gelu | 1.0737x | phase2_n4 |
| L2 | P91 | kernelbench_l2_91_91_convtranspose2d_softmax_biasadd_scaling_sigmoid | 1.0303x | n2 |
| L2 | P92 | kernelbench_l2_92_92_conv2d_groupnorm_tanh_hardswish_residualadd_logsumexp | 1.0000x | root |
| L2 | P93 | kernelbench_l2_93_93_convtranspose2d_add_min_gelu_multiply | 1.7297x | n2 |
| L2 | P94 | kernelbench_l2_94_94_gemm_biasadd_hardtanh_mish_groupnorm | 0.9937x | root |
| L2 | P95 | kernelbench_l2_95_95_matmul_add_swish_tanh_gelu_hardtanh | 1.0552x | n2 |
| L2 | P96 | kernelbench_l2_96_96_convtranspose3d_multiply_max_globalavgpool_clamp | 0.9919x | root |
| L2 | P97 | kernelbench_l2_97_97_matmul_batchnorm_biasadd_divide_swish | 1.0465x | n1 |
| L2 | P98 | kernelbench_l2_98_98_matmul_avgpool_gelu_scale_max | 1.0098x | root |
| L2 | P99 | kernelbench_l2_99_99_matmul_gelu_softmax | 1.0133x | n3 |
| L2 | P100 | kernelbench_l2_100_100_convtranspose3d_clamp_min_divide | N/A | N/A |
| L3 | P1 | kernelbench_l3_1_1_mlp | 1.0194x | phase2_n3 |
| L3 | P2 | kernelbench_l3_2_2_shallowwidemlp | 1.0000x | root |
| L3 | P3 | kernelbench_l3_3_3_deepnarrowmlp | 1.0000x | n4 |
| L3 | P4 | kernelbench_l3_4_4_lenet5 | 0.9804x | root |
| L3 | P5 | kernelbench_l3_5_5_alexnet | 1.0214x | n9 |
| L3 | P6 | kernelbench_l3_6_6_googlenetinceptionmodule | 1.0000x | root |
| L3 | P7 | kernelbench_l3_7_7_googlenetinceptionv1 | 0.9851x | n5 |
| L3 | P8 | kernelbench_l3_8_8_resnetbasicblock | 0.9975x | root |
| L3 | P9 | kernelbench_l3_9_9_resnet18 | 0.8852x | n6 |
| L3 | P10 | kernelbench_l3_10_10_resnet101 | N/A | N/A |
| L3 | P11 | kernelbench_l3_11_11_vgg16 | 1.0039x | n3 |
| L3 | P12 | kernelbench_l3_12_12_vgg19 | 1.0511x | root |
| L3 | P13 | kernelbench_l3_13_13_densenet121transitionlayer | 0.9873x | root |
| L3 | P14 | kernelbench_l3_14_14_densenet121denseblock | 0.9835x | n8 |
| L3 | P15 | kernelbench_l3_15_15_densenet121 | 1.0246x | root |
| L3 | P16 | kernelbench_l3_16_16_densenet201 | 0.8214x | phase2_n4 |
| L3 | P17 | kernelbench_l3_17_17_squeezenetfiremodule | 1.3881x | phase2_n2 |
| L3 | P18 | kernelbench_l3_18_18_squeezenet | N/A | N/A |
| L3 | P19 | kernelbench_l3_19_19_mobilenetv1 | 0.8485x | n5 |
| L3 | P20 | kernelbench_l3_20_20_mobilenetv2 | 0.7686x | n8 |
| L3 | P21 | kernelbench_l3_21_21_efficientnetmbconv | 0.5584x | n3 |
| L3 | P22 | kernelbench_l3_22_22_efficientnetb0 | 0.7571x | n3 |
| L3 | P23 | kernelbench_l3_23_23_efficientnetb1 | N/A | N/A |
| L3 | P24 | kernelbench_l3_24_24_efficientnetb2 | N/A | N/A |
| L3 | P25 | kernelbench_l3_25_25_shufflenetunit | 0.9916x | phase2_n5 |
| L3 | P26 | kernelbench_l3_26_26_shufflenet | N/A | N/A |
| L3 | P27 | kernelbench_l3_27_27_regnet | 0.9873x | n10 |
| L3 | P28 | kernelbench_l3_28_28_visiontransformer | 0.8193x | root |
| L3 | P29 | kernelbench_l3_29_29_swinmlp | N/A | N/A |
| L3 | P30 | kernelbench_l3_30_30_swintransformerv2 | N/A | N/A |
| L3 | P31 | kernelbench_l3_31_31_visionattention | 2.5498x | phase2_n1 |
| L3 | P32 | kernelbench_l3_32_32_convolutionalvisiontransformer | N/A | N/A |
| L3 | P33 | kernelbench_l3_33_33_vanillarnn | 0.9931x | root |
| L3 | P34 | kernelbench_l3_34_34_vanillarnnhidden | 0.8333x | root |
| L3 | P35 | kernelbench_l3_35_35_lstm | 0.9964x | root |
| L3 | P36 | kernelbench_l3_36_36_lstmhn | 0.9927x | phase2_n2 |
| L3 | P37 | kernelbench_l3_37_37_lstmcn | 1.0105x | phase2_n4 |
| L3 | P38 | kernelbench_l3_38_38_lstmbidirectional | 1.0052x | n4 |
| L3 | P39 | kernelbench_l3_39_39_gru | 1.0000x | root |
| L3 | P40 | kernelbench_l3_40_40_gruhidden | 0.9966x | root |
| L3 | P41 | kernelbench_l3_41_41_grubidirectional | 0.9974x | phase2_n3 |
| L3 | P42 | kernelbench_l3_42_42_grubidirectionalhidden | 1.0714x | n4 |
| L3 | P43 | kernelbench_l3_43_43_mingptcausalattention | 2.3340x | phase2_n4 |
| L3 | P44 | kernelbench_l3_44_44_minigptblock | 0.9459x | n5 |
| L3 | P45 | kernelbench_l3_45_45_unetsoftmax | 0.9911x | root |
| L3 | P46 | kernelbench_l3_46_46_netvladwithghostclusters | 0.9943x | root |
| L3 | P47 | kernelbench_l3_47_47_netvladnoghostclusters | 1.0147x | phase2_n5 |
| L3 | P48 | kernelbench_l3_48_48_mamba2returny | 1.0586x | n9 |
| L3 | P49 | kernelbench_l3_49_49_mamba2returnfinalstate | N/A | N/A |
| L3 | P50 | kernelbench_l3_50_50_reluselfattention | 0.9379x | root |
