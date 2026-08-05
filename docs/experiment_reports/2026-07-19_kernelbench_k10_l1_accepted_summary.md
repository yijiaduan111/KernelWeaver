# KernelBench K=10 Accepted Results Summary

Date: 2026-07-19
Source: `/data/dyj/KernelWeaver/runs_final/kernelbench_k10/`
Accepted tasks: 91

## Macro Metrics

| Metric | Value |
|---|---:|
| Compile Rate | 0.9890 |
| Correct Rate | 0.9231 |
| Success@10 | 0.9231 |
| Fast1@10 | 0.8462 |
| Geomean Speedup | 1.3588x |
| Median Speedup | 1.0959x |
| Best Speedup | 9.7517x |
| Mean Speedup | 1.5803x |
| First Correct Median | 1.0 |
| Refine Gain Median | 1.0790x |

## Per-Task Best Correct Speedup

| Level | Problem | Task Name | Best Correct Speedup@10 | Best Node |
|---:|---:|---|---:|---|
| L1 | P1 | kernelbench_l1_1_1_square_matrix_multiplication | 1.4211x | phase2_n5 |
| L1 | P2 | kernelbench_l1_2_2_standard_matrix_multiplication | 1.0043x | phase2_n4 |
| L1 | P3 | kernelbench_l1_3_3_batched_matrix_multiplication | 1.7187x | phase2_n2 |
| L1 | P4 | kernelbench_l1_4_4_matrix_vector_multiplication | 1.0023x | phase2_n5 |
| L1 | P5 | kernelbench_l1_5_5_matrix_scalar_multiplication | 1.0392x | n4 |
| L1 | P6 | kernelbench_l1_6_6_matmul_with_large_k_dimension | 1.0000x | n3 |
| L1 | P7 | kernelbench_l1_7_7_matmul_with_small_k_dimension | 1.2116x | phase2_n3 |
| L1 | P8 | kernelbench_l1_8_8_matmul_with_irregular_shapes | 1.4753x | phase2_n2 |
| L1 | P9 | kernelbench_l1_9_9_tall_skinny_matrix_multiplication | 1.1686x | phase2_n1 |
| L1 | P10 | kernelbench_l1_10_10_3d_tensor_matrix_multiplication | 1.5478x | phase2_n4 |
| L1 | P11 | kernelbench_l1_11_11_4d_tensor_matrix_multiplication | 1.0198x | n1 |
| L1 | P12 | kernelbench_l1_12_12_matmul_with_diagonal_matrices | 1.0068x | phase2_n3 |
| L1 | P13 | kernelbench_l1_13_13_matmul_for_symmetric_matrices | 1.3977x | phase2_n4 |
| L1 | P14 | kernelbench_l1_14_14_matmul_for_upper_triangular_matrices | 1.8963x | n4 |
| L1 | P15 | kernelbench_l1_15_15_matmul_for_lower_triangular_matrices | 0.9959x | phase2_n2 |
| L1 | P16 | kernelbench_l1_16_16_matmul_with_transposed_a | 1.0327x | phase2_n3 |
| L1 | P17 | kernelbench_l1_17_17_matmul_with_transposed_b | 1.0194x | phase2_n2 |
| L1 | P18 | kernelbench_l1_18_18_matmul_with_transposed_both | 0.9957x | n5 |
| L1 | P19 | kernelbench_l1_19_19_relu | 1.0000x | n4 |
| L1 | P20 | kernelbench_l1_20_20_leakyrelu | 1.0175x | n4 |
| L1 | P21 | kernelbench_l1_21_21_sigmoid | 1.1589x | phase2_n1 |
| L1 | P22 | kernelbench_l1_22_22_tanh | 1.0000x | phase2_n3 |
| L1 | P23 | kernelbench_l1_23_23_softmax | 1.3105x | phase2_n4 |
| L1 | P24 | kernelbench_l1_24_24_logsoftmax | 1.2941x | phase2_n2 |
| L1 | P25 | kernelbench_l1_25_25_swish | 2.3448x | phase2_n1 |
| L1 | P26 | kernelbench_l1_26_26_gelu | 1.0132x | phase2_n3 |
| L1 | P27 | kernelbench_l1_27_27_selu | 1.0000x | phase2_n2 |
| L1 | P28 | kernelbench_l1_28_28_hardsigmoid | 1.0066x | phase2_n1 |
| L1 | P29 | kernelbench_l1_29_29_softplus | 1.8377x | n4 |
| L1 | P30 | kernelbench_l1_30_30_softsign | 3.2318x | phase2_n3 |
| L1 | P31 | kernelbench_l1_31_31_elu | 1.0067x | phase2_n5 |
| L1 | P32 | kernelbench_l1_32_32_hardtanh | 1.1250x | phase2_n4 |
| L1 | P33 | kernelbench_l1_33_33_batchnorm | 1.0109x | phase2_n3 |
| L1 | P34 | kernelbench_l1_34_34_instancenorm | N/A | N/A |
| L1 | P35 | kernelbench_l1_35_35_groupnorm | N/A | N/A |
| L1 | P36 | kernelbench_l1_36_36_rmsnorm | 0.7122x | phase2_n5 |
| L1 | P37 | kernelbench_l1_37_37_frobeniusnorm | 1.0633x | phase2_n4 |
| L1 | P38 | kernelbench_l1_38_38_l1norm | N/A | N/A |
| L1 | P39 | kernelbench_l1_39_39_l2norm | N/A | N/A |
| L1 | P40 | kernelbench_l1_40_40_layernorm | 5.5577x | phase2_n3 |
| L1 | P41 | kernelbench_l1_41_41_max_pooling_1d | 2.3108x | phase2_n5 |
| L1 | P42 | kernelbench_l1_42_42_max_pooling_2d | 1.9762x | phase2_n1 |
| L1 | P43 | kernelbench_l1_43_43_max_pooling_3d | 1.2072x | phase2_n1 |
| L1 | P44 | kernelbench_l1_44_44_average_pooling_1d | 1.4413x | phase2_n2 |
| L1 | P45 | kernelbench_l1_45_45_average_pooling_2d | 1.0879x | phase2_n2 |
| L1 | P46 | kernelbench_l1_46_46_average_pooling_3d | 1.0194x | n2 |
| L1 | P47 | kernelbench_l1_47_47_sum_reduction_over_a_dimension | 1.1226x | n3 |
| L1 | P48 | kernelbench_l1_48_48_mean_reduction_over_a_dimension | 1.0031x | phase2_n4 |
| L1 | P49 | kernelbench_l1_49_49_max_reduction_over_a_dimension | 1.1014x | n4 |
| L1 | P50 | kernelbench_l1_50_50_conv_standard_2d_square_input_square_kernel | 1.1685x | phase2_n2 |
| L1 | P51 | kernelbench_l1_51_51_argmax_over_a_dimension | 1.0904x | phase2_n2 |
| L1 | P52 | kernelbench_l1_52_52_argmin_over_a_dimension | 1.0010x | phase2_n2 |
| L1 | P53 | kernelbench_l1_53_53_min_reduction_over_a_dimension | 1.0000x | phase2_n2 |
| L1 | P54 | kernelbench_l1_54_54_conv_standard_3d_square_input_square_kernel | 0.9964x | n3 |
| L1 | P55 | kernelbench_l1_55_55_conv_standard_2d_asymmetric_input_square_kernel | 1.0190x | phase2_n5 |
| L1 | P56 | kernelbench_l1_56_56_conv_standard_2d_asymmetric_input_asymmetric_kernel | 1.0651x | phase2_n1 |
| L1 | P57 | kernelbench_l1_57_57_conv_transposed_2d_square_input_square_kernel | 1.0000x | n2 |
| L1 | P61 | kernelbench_l1_61_61_conv_transposed_3d_square_input_square_kernel | 1.0000x | phase2_n1 |
| L1 | P65 | kernelbench_l1_65_65_conv_transposed_2d_square_input_asymmetric_kernel | 1.0233x | phase2_n4 |
| L1 | P66 | kernelbench_l1_66_66_conv_standard_3d_asymmetric_input_asymmetric_kernel | 1.0455x | n2 |
| L1 | P70 | kernelbench_l1_70_70_conv_transposed_3d_asymmetric_input_square_kernel | N/A | N/A |
| L1 | P71 | kernelbench_l1_71_71_conv_transposed_2d_asymmetric_input_square_kernel | 0.9949x | phase2_n3 |
| L1 | P72 | kernelbench_l1_72_72_conv_transposed_3d_asymmetric_input_asymmetric_kernel_strided_padded_grouped | 2.4882x | phase2_n1 |
| L1 | P73 | kernelbench_l1_73_73_conv_transposed_3d_asymmetric_input_square_kernel_strided_padded_grouped | 1.0055x | n1 |
| L1 | P74 | kernelbench_l1_74_74_conv_transposed_1d_dilated | 1.0428x | n3 |
| L1 | P75 | kernelbench_l1_75_75_conv_transposed_2d_asymmetric_input_asymmetric_kernel_strided_grouped_padded_dilated | 3.0178x | phase2_n5 |
| L1 | P76 | kernelbench_l1_76_76_conv_standard_1d_dilated_strided | 1.1751x | phase2_n1 |
| L1 | P77 | kernelbench_l1_77_77_conv_transposed_3d_square_input_square_kernel_padded_dilated_strided | 0.9989x | phase2_n2 |
| L1 | P78 | kernelbench_l1_78_78_conv_transposed_2d_asymmetric_input_asymmetric_kernel_padded | 1.0000x | n2 |
| L1 | P79 | kernelbench_l1_79_79_conv_transposed_1d_asymmetric_input_square_kernel_padded_strided_dilated | 0.9980x | phase2_n5 |
| L1 | P80 | kernelbench_l1_80_80_conv_standard_2d_square_input_asymmetric_kernel_dilated_padded | 1.1040x | phase2_n5 |
| L1 | P81 | kernelbench_l1_81_81_conv_transposed_2d_asymmetric_input_square_kernel_dilated_padded_strided | 1.6818x | n5 |
| L1 | P82 | kernelbench_l1_82_82_conv_depthwise_2d_square_input_square_kernel | 1.2460x | phase2_n5 |
| L1 | P83 | kernelbench_l1_83_83_conv_depthwise_2d_square_input_asymmetric_kernel | 1.7323x | n2 |
| L1 | P84 | kernelbench_l1_84_84_conv_depthwise_2d_asymmetric_input_square_kernel | 1.2841x | n3 |
| L1 | P85 | kernelbench_l1_85_85_conv_depthwise_2d_asymmetric_input_asymmetric_kernel | 1.6767x | phase2_n2 |
| L1 | P86 | kernelbench_l1_86_86_conv_depthwise_separable_2d | 1.0119x | n5 |
| L1 | P87 | kernelbench_l1_87_87_conv_pointwise_2d | N/A | N/A |
| L1 | P88 | kernelbench_l1_88_88_mingptnewgelu | 9.7517x | n5 |
| L1 | P89 | kernelbench_l1_89_89_cumsum | 1.1739x | phase2_n5 |
| L1 | P90 | kernelbench_l1_90_90_cumprod | 1.1961x | phase2_n4 |
| L1 | P91 | kernelbench_l1_91_91_cumsum_reverse | 1.0033x | n3 |
| L1 | P92 | kernelbench_l1_92_92_cumsum_exclusive | 3.3000x | n8 |
| L1 | P93 | kernelbench_l1_93_93_masked_cumsum | 1.0157x | phase2_n3 |
| L1 | P94 | kernelbench_l1_94_94_mseloss | 3.1023x | phase2_n2 |
| L1 | P95 | kernelbench_l1_95_95_crossentropyloss | N/A | N/A |
| L1 | P96 | kernelbench_l1_96_96_huberloss | 2.1705x | n9 |
| L1 | P97 | kernelbench_l1_97_97_scaleddotproductattention | 2.1752x | n8 |
| L1 | P98 | kernelbench_l1_98_98_kldivloss | 5.6000x | n4 |
| L1 | P99 | kernelbench_l1_99_99_tripletmarginloss | 4.0719x | n1 |
| L1 | P100 | kernelbench_l1_100_100_hingeloss | 2.4015x | n4 |
