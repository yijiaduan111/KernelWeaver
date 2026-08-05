# Direct Claude KernelBench Snapshot



## Summary

- Accepted tasks: `250/250`
- Level 1: `100/100`
- Level 2: `100/100`
- Level 3: `50/50`
- Missing tasks: none
- Note: this direct baseline generates one candidate per task, so `Success@10` and `Fast1@10` collapse to single-sample outcomes.

## Metrics by Level

| Metric | Overall | L1 | L2 | L3 |
|---|---:|---:|---:|---:|
| Compile Rate | 0.8520 | 0.8000 | 0.9300 | 0.8000 |
| Correct Rate | 0.6600 | 0.5200 | 0.8300 | 0.6000 |
| Success@10 | 0.6600 | 0.5200 | 0.8300 | 0.6000 |
| Fast1@10 | 0.3440 | 0.1600 | 0.6000 | 0.2000 |
| Geomean Speedup | 0.7678x | 0.8187x | 0.7361x | 0.7718x |
| Median Speedup | 1.0000x | 0.9531x | 1.0309x | 0.9814x |
| Best Speedup | 8.7436x | 8.7436x | 2.6701x | 2.3168x |
| Mean Speedup | 1.0624x | 1.1535x | 1.0438x | 0.9558x |
| First Correct Median | 1 | 1 | 1 | 1 |
| Refine Gain Median | 1.0000x | 1.0000x | 1.0000x | 1.0000x |

## Per-Task Best Correct Speedup

| Level | Problem | Task Name | Best Correct Speedup | Best Node |
|---:|---:|---|---:|---|
| L1 | P1 | kernelbench_l1_1_1_square_matrix_multiplication | N/A | N/A |
| L1 | P2 | kernelbench_l1_2_2_standard_matrix_multiplication | N/A | N/A |
| L1 | P3 | kernelbench_l1_3_3_batched_matrix_multiplication | 0.9936x | candidate_1 |
| L1 | P4 | kernelbench_l1_4_4_matrix_vector_multiplication | N/A | N/A |
| L1 | P5 | kernelbench_l1_5_5_matrix_scalar_multiplication | 0.9649x | candidate_1 |
| L1 | P6 | kernelbench_l1_6_6_matmul_with_large_k_dimension | 0.0761x | candidate_1 |
| L1 | P7 | kernelbench_l1_7_7_matmul_with_small_k_dimension | 0.8045x | candidate_1 |
| L1 | P8 | kernelbench_l1_8_8_matmul_with_irregular_shapes | N/A | N/A |
| L1 | P9 | kernelbench_l1_9_9_tall_skinny_matrix_multiplication | N/A | N/A |
| L1 | P10 | kernelbench_l1_10_10_3d_tensor_matrix_multiplication | 0.1236x | candidate_1 |
| L1 | P11 | kernelbench_l1_11_11_4d_tensor_matrix_multiplication | N/A | N/A |
| L1 | P12 | kernelbench_l1_12_12_matmul_with_diagonal_matrices | N/A | N/A |
| L1 | P13 | kernelbench_l1_13_13_matmul_for_symmetric_matrices | N/A | N/A |
| L1 | P14 | kernelbench_l1_14_14_matmul_for_upper_triangular_matrices | 0.4611x | candidate_1 |
| L1 | P15 | kernelbench_l1_15_15_matmul_for_lower_triangular_matrices | 0.3918x | candidate_1 |
| L1 | P16 | kernelbench_l1_16_16_matmul_with_transposed_a | 0.0783x | candidate_1 |
| L1 | P17 | kernelbench_l1_17_17_matmul_with_transposed_b | N/A | N/A |
| L1 | P18 | kernelbench_l1_18_18_matmul_with_transposed_both | 0.0792x | candidate_1 |
| L1 | P19 | kernelbench_l1_19_19_relu | 0.8901x | candidate_1 |
| L1 | P20 | kernelbench_l1_20_20_leakyrelu | N/A | N/A |
| L1 | P21 | kernelbench_l1_21_21_sigmoid | 0.9759x | candidate_1 |
| L1 | P22 | kernelbench_l1_22_22_tanh | 0.6814x | candidate_1 |
| L1 | P23 | kernelbench_l1_23_23_softmax | 0.6642x | candidate_1 |
| L1 | P24 | kernelbench_l1_24_24_logsoftmax | 0.8835x | candidate_1 |
| L1 | P25 | kernelbench_l1_25_25_swish | 2.2784x | candidate_1 |
| L1 | P26 | kernelbench_l1_26_26_gelu | 0.6452x | candidate_1 |
| L1 | P27 | kernelbench_l1_27_27_selu | N/A | N/A |
| L1 | P28 | kernelbench_l1_28_28_hardsigmoid | 0.6829x | candidate_1 |
| L1 | P29 | kernelbench_l1_29_29_softplus | 0.6699x | candidate_1 |
| L1 | P30 | kernelbench_l1_30_30_softsign | 3.2184x | candidate_1 |
| L1 | P31 | kernelbench_l1_31_31_elu | N/A | N/A |
| L1 | P32 | kernelbench_l1_32_32_hardtanh | 0.6573x | candidate_1 |
| L1 | P33 | kernelbench_l1_33_33_batchnorm | N/A | N/A |
| L1 | P34 | kernelbench_l1_34_34_instancenorm | N/A | N/A |
| L1 | P35 | kernelbench_l1_35_35_groupnorm | N/A | N/A |
| L1 | P36 | kernelbench_l1_36_36_rmsnorm | N/A | N/A |
| L1 | P37 | kernelbench_l1_37_37_frobeniusnorm | N/A | N/A |
| L1 | P38 | kernelbench_l1_38_38_l1norm | N/A | N/A |
| L1 | P39 | kernelbench_l1_39_39_l2norm | N/A | N/A |
| L1 | P40 | kernelbench_l1_40_40_layernorm | 0.6205x | candidate_1 |
| L1 | P41 | kernelbench_l1_41_41_max_pooling_1d | 1.9456x | candidate_1 |
| L1 | P42 | kernelbench_l1_42_42_max_pooling_2d | 1.5223x | candidate_1 |
| L1 | P43 | kernelbench_l1_43_43_max_pooling_3d | N/A | N/A |
| L1 | P44 | kernelbench_l1_44_44_average_pooling_1d | N/A | N/A |
| L1 | P45 | kernelbench_l1_45_45_average_pooling_2d | N/A | N/A |
| L1 | P46 | kernelbench_l1_46_46_average_pooling_3d | N/A | N/A |
| L1 | P47 | kernelbench_l1_47_47_sum_reduction_over_a_dimension | 0.1239x | candidate_1 |
| L1 | P48 | kernelbench_l1_48_48_mean_reduction_over_a_dimension | 0.9297x | candidate_1 |
| L1 | P49 | kernelbench_l1_49_49_max_reduction_over_a_dimension | 0.9225x | candidate_1 |
| L1 | P50 | kernelbench_l1_50_50_conv_standard_2d_square_input_square_kernel | N/A | N/A |
| L1 | P51 | kernelbench_l1_51_51_argmax_over_a_dimension | 0.9297x | candidate_1 |
| L1 | P52 | kernelbench_l1_52_52_argmin_over_a_dimension | 0.8646x | candidate_1 |
| L1 | P53 | kernelbench_l1_53_53_min_reduction_over_a_dimension | 0.8637x | candidate_1 |
| L1 | P54 | kernelbench_l1_54_54_conv_standard_3d_square_input_square_kernel | N/A | N/A |
| L1 | P55 | kernelbench_l1_55_55_conv_standard_2d_asymmetric_input_square_kernel | 1.0000x | candidate_1 |
| L1 | P56 | kernelbench_l1_56_56_conv_standard_2d_asymmetric_input_asymmetric_kernel | N/A | N/A |
| L1 | P57 | kernelbench_l1_57_57_conv_transposed_2d_square_input_square_kernel | N/A | N/A |
| L1 | P58 | kernelbench_l1_58_58_conv_transposed_3d_asymmetric_input_asymmetric_kernel | 1.0109x | candidate_1 |
| L1 | P59 | kernelbench_l1_59_59_conv_standard_3d_asymmetric_input_square_kernel | 0.9730x | candidate_1 |
| L1 | P60 | kernelbench_l1_60_60_conv_standard_3d_square_input_asymmetric_kernel | 1.0030x | candidate_1 |
| L1 | P61 | kernelbench_l1_61_61_conv_transposed_3d_square_input_square_kernel | 1.0000x | candidate_1 |
| L1 | P62 | kernelbench_l1_62_62_conv_standard_2d_square_input_asymmetric_kernel | N/A | N/A |
| L1 | P63 | kernelbench_l1_63_63_conv_standard_2d_square_input_square_kernel | N/A | N/A |
| L1 | P64 | kernelbench_l1_64_64_conv_transposed_1d | N/A | N/A |
| L1 | P65 | kernelbench_l1_65_65_conv_transposed_2d_square_input_asymmetric_kernel | 0.9484x | candidate_1 |
| L1 | P66 | kernelbench_l1_66_66_conv_standard_3d_asymmetric_input_asymmetric_kernel | N/A | N/A |
| L1 | P67 | kernelbench_l1_67_67_conv_standard_1d | N/A | N/A |
| L1 | P68 | kernelbench_l1_68_68_conv_transposed_3d_square_input_asymmetric_kernel | 0.9969x | candidate_1 |
| L1 | P69 | kernelbench_l1_69_69_conv_transposed_2d_asymmetric_input_asymmetric_kernel | N/A | N/A |
| L1 | P70 | kernelbench_l1_70_70_conv_transposed_3d_asymmetric_input_square_kernel | 1.0026x | candidate_1 |
| L1 | P71 | kernelbench_l1_71_71_conv_transposed_2d_asymmetric_input_square_kernel | N/A | N/A |
| L1 | P72 | kernelbench_l1_72_72_conv_transposed_3d_asymmetric_input_asymmetric_kernel_strided_padded_grouped | 0.9570x | candidate_1 |
| L1 | P73 | kernelbench_l1_73_73_conv_transposed_3d_asymmetric_input_square_kernel_strided_padded_grouped | 0.9788x | candidate_1 |
| L1 | P74 | kernelbench_l1_74_74_conv_transposed_1d_dilated | N/A | N/A |
| L1 | P75 | kernelbench_l1_75_75_conv_transposed_2d_asymmetric_input_asymmetric_kernel_strided_grouped_padded_dilated | 0.9636x | candidate_1 |
| L1 | P76 | kernelbench_l1_76_76_conv_standard_1d_dilated_strided | N/A | N/A |
| L1 | P77 | kernelbench_l1_77_77_conv_transposed_3d_square_input_square_kernel_padded_dilated_strided | 0.9923x | candidate_1 |
| L1 | P78 | kernelbench_l1_78_78_conv_transposed_2d_asymmetric_input_asymmetric_kernel_padded | N/A | N/A |
| L1 | P79 | kernelbench_l1_79_79_conv_transposed_1d_asymmetric_input_square_kernel_padded_strided_dilated | 0.9670x | candidate_1 |
| L1 | P80 | kernelbench_l1_80_80_conv_standard_2d_square_input_asymmetric_kernel_dilated_padded | N/A | N/A |
| L1 | P81 | kernelbench_l1_81_81_conv_transposed_2d_asymmetric_input_square_kernel_dilated_padded_strided | N/A | N/A |
| L1 | P82 | kernelbench_l1_82_82_conv_depthwise_2d_square_input_square_kernel | N/A | N/A |
| L1 | P83 | kernelbench_l1_83_83_conv_depthwise_2d_square_input_asymmetric_kernel | N/A | N/A |
| L1 | P84 | kernelbench_l1_84_84_conv_depthwise_2d_asymmetric_input_square_kernel | N/A | N/A |
| L1 | P85 | kernelbench_l1_85_85_conv_depthwise_2d_asymmetric_input_asymmetric_kernel | 1.2174x | candidate_1 |
| L1 | P86 | kernelbench_l1_86_86_conv_depthwise_separable_2d | 0.9493x | candidate_1 |
| L1 | P87 | kernelbench_l1_87_87_conv_pointwise_2d | N/A | N/A |
| L1 | P88 | kernelbench_l1_88_88_mingptnewgelu | 8.7436x | candidate_1 |
| L1 | P89 | kernelbench_l1_89_89_cumsum | 0.3674x | candidate_1 |
| L1 | P90 | kernelbench_l1_90_90_cumprod | 0.6465x | candidate_1 |
| L1 | P91 | kernelbench_l1_91_91_cumsum_reverse | 1.1633x | candidate_1 |
| L1 | P92 | kernelbench_l1_92_92_cumsum_exclusive | 1.5833x | candidate_1 |
| L1 | P93 | kernelbench_l1_93_93_masked_cumsum | N/A | N/A |
| L1 | P94 | kernelbench_l1_94_94_mseloss | 2.8933x | candidate_1 |
| L1 | P95 | kernelbench_l1_95_95_crossentropyloss | N/A | N/A |
| L1 | P96 | kernelbench_l1_96_96_huberloss | 2.0190x | candidate_1 |
| L1 | P97 | kernelbench_l1_97_97_scaleddotproductattention | N/A | N/A |
| L1 | P98 | kernelbench_l1_98_98_kldivloss | 2.6638x | candidate_1 |
| L1 | P99 | kernelbench_l1_99_99_tripletmarginloss | N/A | N/A |
| L1 | P100 | kernelbench_l1_100_100_hingeloss | N/A | N/A |
| L2 | P1 | kernelbench_l2_1_1_conv2d_relu_biasadd | 1.2404x | candidate_1 |
| L2 | P2 | kernelbench_l2_2_2_convtranspose2d_biasadd_clamp_scaling_clamp_divide | 1.9000x | candidate_1 |
| L2 | P3 | kernelbench_l2_3_3_convtranspose3d_sum_layernorm_avgpool_gelu | N/A | N/A |
| L2 | P4 | kernelbench_l2_4_4_conv2d_mish_mish | 1.0038x | candidate_1 |
| L2 | P5 | kernelbench_l2_5_5_convtranspose2d_subtract_tanh | 1.2203x | candidate_1 |
| L2 | P6 | kernelbench_l2_6_6_conv3d_softmax_maxpool_maxpool | 1.0877x | candidate_1 |
| L2 | P7 | kernelbench_l2_7_7_conv3d_relu_leakyrelu_gelu_sigmoid_biasadd | 1.9688x | candidate_1 |
| L2 | P8 | kernelbench_l2_8_8_conv3d_divide_max_globalavgpool_biasadd_sum | N/A | N/A |
| L2 | P9 | kernelbench_l2_9_9_matmul_subtract_multiply_relu | 0.0271x | candidate_1 |
| L2 | P10 | kernelbench_l2_10_10_convtranspose2d_maxpool_hardtanh_mean_tanh | 1.0000x | candidate_1 |
| L2 | P11 | kernelbench_l2_11_11_convtranspose2d_batchnorm_tanh_maxpool_groupnorm | 1.1101x | candidate_1 |
| L2 | P12 | kernelbench_l2_12_12_gemm_multiply_leakyrelu | 0.0272x | candidate_1 |
| L2 | P13 | kernelbench_l2_13_13_convtranspose3d_mean_add_softmax_tanh_scaling | 0.6547x | candidate_1 |
| L2 | P14 | kernelbench_l2_14_14_gemm_divide_sum_scaling | N/A | N/A |
| L2 | P15 | kernelbench_l2_15_15_convtranspose3d_batchnorm_subtract | 0.9742x | candidate_1 |
| L2 | P16 | kernelbench_l2_16_16_convtranspose2d_mish_add_hardtanh_scaling | N/A | N/A |
| L2 | P17 | kernelbench_l2_17_17_conv2d_instancenorm_divide | N/A | N/A |
| L2 | P18 | kernelbench_l2_18_18_matmul_sum_max_avgpool_logsumexp_logsumexp | 1.0000x | candidate_1 |
| L2 | P19 | kernelbench_l2_19_19_convtranspose2d_gelu_groupnorm | N/A | N/A |
| L2 | P20 | kernelbench_l2_20_20_convtranspose3d_sum_residualadd_multiply_residualadd | 1.5962x | candidate_1 |
| L2 | P21 | kernelbench_l2_21_21_conv2d_add_scale_sigmoid_groupnorm | 1.4038x | candidate_1 |
| L2 | P22 | kernelbench_l2_22_22_matmul_scale_residualadd_clamp_logsumexp_mish | 1.0439x | candidate_1 |
| L2 | P23 | kernelbench_l2_23_23_conv3d_groupnorm_mean | 1.3116x | candidate_1 |
| L2 | P24 | kernelbench_l2_24_24_conv3d_min_softmax | 1.0000x | candidate_1 |
| L2 | P25 | kernelbench_l2_25_25_conv2d_min_tanh_tanh | 0.9932x | candidate_1 |
| L2 | P26 | kernelbench_l2_26_26_convtranspose3d_add_hardswish | 1.2535x | candidate_1 |
| L2 | P27 | kernelbench_l2_27_27_conv3d_hardswish_groupnorm_mean | 1.3189x | candidate_1 |
| L2 | P28 | kernelbench_l2_28_28_bmm_instancenorm_sum_residualadd_multiply | N/A | N/A |
| L2 | P29 | kernelbench_l2_29_29_matmul_mish_mish | 1.0033x | candidate_1 |
| L2 | P30 | kernelbench_l2_30_30_gemm_groupnorm_hardtanh | 1.0548x | candidate_1 |
| L2 | P31 | kernelbench_l2_31_31_conv2d_min_add_multiply | 1.4904x | candidate_1 |
| L2 | P32 | kernelbench_l2_32_32_conv2d_scaling_min | 1.2573x | candidate_1 |
| L2 | P33 | kernelbench_l2_33_33_gemm_scale_batchnorm | 0.0271x | candidate_1 |
| L2 | P34 | kernelbench_l2_34_34_convtranspose3d_layernorm_gelu_scaling | N/A | N/A |
| L2 | P35 | kernelbench_l2_35_35_conv2d_subtract_hardswish_maxpool_mish | N/A | N/A |
| L2 | P36 | kernelbench_l2_36_36_convtranspose2d_min_sum_gelu_add | 0.8804x | candidate_1 |
| L2 | P37 | kernelbench_l2_37_37_matmul_swish_sum_groupnorm | 0.0568x | candidate_1 |
| L2 | P38 | kernelbench_l2_38_38_convtranspose3d_avgpool_clamp_softmax_multiply | 0.9964x | candidate_1 |
| L2 | P39 | kernelbench_l2_39_39_gemm_scale_batchnorm | 0.8954x | candidate_1 |
| L2 | P40 | kernelbench_l2_40_40_matmul_scaling_residualadd | 1.0692x | candidate_1 |
| L2 | P41 | kernelbench_l2_41_41_gemm_batchnorm_gelu_relu | 1.0000x | candidate_1 |
| L2 | P42 | kernelbench_l2_42_42_convtranspose2d_globalavgpool_biasadd_logsumexp_sum_multiply | 1.0000x | candidate_1 |
| L2 | P43 | kernelbench_l2_43_43_conv3d_max_logsumexp_relu | 1.0080x | candidate_1 |
| L2 | P44 | kernelbench_l2_44_44_convtranspose2d_multiply_globalavgpool_globalavgpool_mean | 0.9834x | candidate_1 |
| L2 | P45 | kernelbench_l2_45_45_gemm_sigmoid_logsumexp | 0.0071x | candidate_1 |
| L2 | P46 | kernelbench_l2_46_46_conv2d_subtract_tanh_subtract_avgpool | 1.4132x | candidate_1 |
| L2 | P47 | kernelbench_l2_47_47_conv3d_mish_tanh | 1.1770x | candidate_1 |
| L2 | P48 | kernelbench_l2_48_48_conv3d_scaling_tanh_multiply_sigmoid | N/A | N/A |
| L2 | P49 | kernelbench_l2_49_49_convtranspose3d_softmax_sigmoid | 1.0813x | candidate_1 |
| L2 | P50 | kernelbench_l2_50_50_convtranspose3d_scaling_avgpool_biasadd_scaling | 0.9956x | candidate_1 |
| L2 | P51 | kernelbench_l2_51_51_gemm_subtract_globalavgpool_logsumexp_gelu_residualadd | 1.0369x | candidate_1 |
| L2 | P52 | kernelbench_l2_52_52_conv2d_activation_batchnorm | 1.3341x | candidate_1 |
| L2 | P53 | kernelbench_l2_53_53_gemm_scaling_hardtanh_gelu | 1.0354x | candidate_1 |
| L2 | P54 | kernelbench_l2_54_54_conv2d_multiply_leakyrelu_gelu | N/A | N/A |
| L2 | P55 | kernelbench_l2_55_55_matmul_maxpool_sum_scale | 1.0164x | candidate_1 |
| L2 | P56 | kernelbench_l2_56_56_matmul_sigmoid_sum | 0.0248x | candidate_1 |
| L2 | P57 | kernelbench_l2_57_57_conv2d_relu_hardswish | 2.6701x | candidate_1 |
| L2 | P58 | kernelbench_l2_58_58_convtranspose3d_logsumexp_hardswish_subtract_clamp | 1.3756x | candidate_1 |
| L2 | P59 | kernelbench_l2_59_59_matmul_swish_scaling | 0.0283x | candidate_1 |
| L2 | P60 | kernelbench_l2_60_60_convtranspose3d_swish_groupnorm_hardswish | 1.2459x | candidate_1 |
| L2 | P61 | kernelbench_l2_61_61_convtranspose3d_relu_groupnorm | 0.9418x | candidate_1 |
| L2 | P62 | kernelbench_l2_62_62_matmul_groupnorm_leakyrelu_sum | 1.0253x | candidate_1 |
| L2 | P63 | kernelbench_l2_63_63_gemm_relu_divide | 0.0272x | candidate_1 |
| L2 | P64 | kernelbench_l2_64_64_gemm_logsumexp_leakyrelu_leakyrelu_gelu_gelu | 1.0135x | candidate_1 |
| L2 | P65 | kernelbench_l2_65_65_conv2d_avgpool_sigmoid_sum | 0.0716x | candidate_1 |
| L2 | P66 | kernelbench_l2_66_66_matmul_dropout_softmax | N/A | N/A |
| L2 | P67 | kernelbench_l2_67_67_conv2d_gelu_globalavgpool | N/A | N/A |
| L2 | P68 | kernelbench_l2_68_68_matmul_min_subtract | 0.9677x | candidate_1 |
| L2 | P69 | kernelbench_l2_69_69_conv2d_hardswish_relu | 1.3344x | candidate_1 |
| L2 | P70 | kernelbench_l2_70_70_gemm_sigmoid_scaling_residualadd | 0.0279x | candidate_1 |
| L2 | P71 | kernelbench_l2_71_71_conv2d_divide_leakyrelu | 1.3739x | candidate_1 |
| L2 | P72 | kernelbench_l2_72_72_convtranspose3d_batchnorm_avgpool_avgpool | 0.9966x | candidate_1 |
| L2 | P73 | kernelbench_l2_73_73_conv2d_batchnorm_scaling | 0.9867x | candidate_1 |
| L2 | P74 | kernelbench_l2_74_74_convtranspose3d_leakyrelu_multiply_leakyrelu_max | 1.3053x | candidate_1 |
| L2 | P75 | kernelbench_l2_75_75_gemm_groupnorm_min_biasadd | N/A | N/A |
| L2 | P76 | kernelbench_l2_76_76_gemm_add_relu | 1.0063x | candidate_1 |
| L2 | P77 | kernelbench_l2_77_77_convtranspose3d_scale_batchnorm_globalavgpool | 1.0000x | candidate_1 |
| L2 | P78 | kernelbench_l2_78_78_convtranspose3d_max_max_sum | 0.9960x | candidate_1 |
| L2 | P79 | kernelbench_l2_79_79_conv3d_multiply_instancenorm_clamp_multiply_max | N/A | N/A |
| L2 | P80 | kernelbench_l2_80_80_gemm_max_subtract_gelu | 1.0067x | candidate_1 |
| L2 | P81 | kernelbench_l2_81_81_gemm_swish_divide_clamp_tanh_clamp | 1.0503x | candidate_1 |
| L2 | P82 | kernelbench_l2_82_82_conv2d_tanh_scaling_biasadd_max | 1.2535x | candidate_1 |
| L2 | P83 | kernelbench_l2_83_83_conv3d_groupnorm_min_clamp_dropout | 1.1638x | candidate_1 |
| L2 | P84 | kernelbench_l2_84_84_gemm_batchnorm_scaling_softmax | 1.0000x | candidate_1 |
| L2 | P85 | kernelbench_l2_85_85_conv2d_groupnorm_scale_maxpool_clamp | 1.2875x | candidate_1 |
| L2 | P86 | kernelbench_l2_86_86_matmul_divide_gelu | 1.0000x | candidate_1 |
| L2 | P87 | kernelbench_l2_87_87_conv2d_subtract_subtract_mish | 1.7250x | candidate_1 |
| L2 | P88 | kernelbench_l2_88_88_gemm_groupnorm_swish_multiply_swish | 1.0689x | candidate_1 |
| L2 | P89 | kernelbench_l2_89_89_convtranspose3d_maxpool_softmax_subtract_swish_max | 1.0694x | candidate_1 |
| L2 | P90 | kernelbench_l2_90_90_conv3d_leakyrelu_sum_clamp_gelu | 1.6453x | candidate_1 |
| L2 | P91 | kernelbench_l2_91_91_convtranspose2d_softmax_biasadd_scaling_sigmoid | 1.0309x | candidate_1 |
| L2 | P92 | kernelbench_l2_92_92_conv2d_groupnorm_tanh_hardswish_residualadd_logsumexp | 1.8525x | candidate_1 |
| L2 | P93 | kernelbench_l2_93_93_convtranspose2d_add_min_gelu_multiply | 1.7111x | candidate_1 |
| L2 | P94 | kernelbench_l2_94_94_gemm_biasadd_hardtanh_mish_groupnorm | N/A | N/A |
| L2 | P95 | kernelbench_l2_95_95_matmul_add_swish_tanh_gelu_hardtanh | N/A | N/A |
| L2 | P96 | kernelbench_l2_96_96_convtranspose3d_multiply_max_globalavgpool_clamp | 1.1165x | candidate_1 |
| L2 | P97 | kernelbench_l2_97_97_matmul_batchnorm_biasadd_divide_swish | 1.0331x | candidate_1 |
| L2 | P98 | kernelbench_l2_98_98_matmul_avgpool_gelu_scale_max | 1.0137x | candidate_1 |
| L2 | P99 | kernelbench_l2_99_99_matmul_gelu_softmax | 1.0000x | candidate_1 |
| L2 | P100 | kernelbench_l2_100_100_convtranspose3d_clamp_min_divide | 1.2358x | candidate_1 |
| L3 | P1 | kernelbench_l3_1_1_mlp | 0.0346x | candidate_1 |
| L3 | P2 | kernelbench_l3_2_2_shallowwidemlp | 0.0364x | candidate_1 |
| L3 | P3 | kernelbench_l3_3_3_deepnarrowmlp | N/A | N/A |
| L3 | P4 | kernelbench_l3_4_4_lenet5 | 0.9631x | candidate_1 |
| L3 | P5 | kernelbench_l3_5_5_alexnet | N/A | N/A |
| L3 | P6 | kernelbench_l3_6_6_googlenetinceptionmodule | 0.9915x | candidate_1 |
| L3 | P7 | kernelbench_l3_7_7_googlenetinceptionv1 | N/A | N/A |
| L3 | P8 | kernelbench_l3_8_8_resnetbasicblock | 1.0675x | candidate_1 |
| L3 | P9 | kernelbench_l3_9_9_resnet18 | N/A | N/A |
| L3 | P10 | kernelbench_l3_10_10_resnet101 | N/A | N/A |
| L3 | P11 | kernelbench_l3_11_11_vgg16 | 0.9732x | candidate_1 |
| L3 | P12 | kernelbench_l3_12_12_vgg19 | 0.9800x | candidate_1 |
| L3 | P13 | kernelbench_l3_13_13_densenet121transitionlayer | N/A | N/A |
| L3 | P14 | kernelbench_l3_14_14_densenet121denseblock | 1.0000x | candidate_1 |
| L3 | P15 | kernelbench_l3_15_15_densenet121 | 0.7296x | candidate_1 |
| L3 | P16 | kernelbench_l3_16_16_densenet201 | 1.0303x | candidate_1 |
| L3 | P17 | kernelbench_l3_17_17_squeezenetfiremodule | 0.9828x | candidate_1 |
| L3 | P18 | kernelbench_l3_18_18_squeezenet | N/A | N/A |
| L3 | P19 | kernelbench_l3_19_19_mobilenetv1 | N/A | N/A |
| L3 | P20 | kernelbench_l3_20_20_mobilenetv2 | N/A | N/A |
| L3 | P21 | kernelbench_l3_21_21_efficientnetmbconv | 0.7861x | candidate_1 |
| L3 | P22 | kernelbench_l3_22_22_efficientnetb0 | 0.7944x | candidate_1 |
| L3 | P23 | kernelbench_l3_23_23_efficientnetb1 | N/A | N/A |
| L3 | P24 | kernelbench_l3_24_24_efficientnetb2 | N/A | N/A |
| L3 | P25 | kernelbench_l3_25_25_shufflenetunit | 0.9957x | candidate_1 |
| L3 | P26 | kernelbench_l3_26_26_shufflenet | 0.9635x | candidate_1 |
| L3 | P27 | kernelbench_l3_27_27_regnet | 1.1301x | candidate_1 |
| L3 | P28 | kernelbench_l3_28_28_visiontransformer | N/A | N/A |
| L3 | P29 | kernelbench_l3_29_29_swinmlp | N/A | N/A |
| L3 | P30 | kernelbench_l3_30_30_swintransformerv2 | 0.8239x | candidate_1 |
| L3 | P31 | kernelbench_l3_31_31_visionattention | N/A | N/A |
| L3 | P32 | kernelbench_l3_32_32_convolutionalvisiontransformer | N/A | N/A |
| L3 | P33 | kernelbench_l3_33_33_vanillarnn | 0.9066x | candidate_1 |
| L3 | P34 | kernelbench_l3_34_34_vanillarnnhidden | 0.8346x | candidate_1 |
| L3 | P35 | kernelbench_l3_35_35_lstm | 0.9867x | candidate_1 |
| L3 | P36 | kernelbench_l3_36_36_lstmhn | 0.9498x | candidate_1 |
| L3 | P37 | kernelbench_l3_37_37_lstmcn | N/A | N/A |
| L3 | P38 | kernelbench_l3_38_38_lstmbidirectional | 1.1201x | candidate_1 |
| L3 | P39 | kernelbench_l3_39_39_gru | N/A | N/A |
| L3 | P40 | kernelbench_l3_40_40_gruhidden | N/A | N/A |
| L3 | P41 | kernelbench_l3_41_41_grubidirectional | 1.0442x | candidate_1 |
| L3 | P42 | kernelbench_l3_42_42_grubidirectionalhidden | 1.0797x | candidate_1 |
| L3 | P43 | kernelbench_l3_43_43_mingptcausalattention | 2.3168x | candidate_1 |
| L3 | P44 | kernelbench_l3_44_44_minigptblock | 1.8451x | candidate_1 |
| L3 | P45 | kernelbench_l3_45_45_unetsoftmax | 0.9563x | candidate_1 |
| L3 | P46 | kernelbench_l3_46_46_netvladwithghostclusters | 0.1977x | candidate_1 |
| L3 | P47 | kernelbench_l3_47_47_netvladnoghostclusters | N/A | N/A |
| L3 | P48 | kernelbench_l3_48_48_mamba2returny | 0.9957x | candidate_1 |
| L3 | P49 | kernelbench_l3_49_49_mamba2returnfinalstate | 1.1586x | candidate_1 |
| L3 | P50 | kernelbench_l3_50_50_reluselfattention | N/A | N/A |

