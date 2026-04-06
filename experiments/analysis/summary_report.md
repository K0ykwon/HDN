# Experiment Summary

## hyperpartisan

| run | acc | loss | depth | throughput | depth corr |
| --- | ---: | ---: | ---: | ---: | ---: |
| twr_gmc8c2_d576_hyperpartisan_100ep | 0.5825 | 0.7744 | 4.933 | 799.3 | 0.9070 |
| twr_gmc8c2_d960_hyperpartisan_100ep | 0.5825 | 1.0874 | 3.784 | 346.3 | 0.7857 |
| twr_gmc8c2_d320_hyperpartisan_100ep | 0.5675 | 0.8158 | 4.378 | 1639.9 | -0.5366 |
| twr_gmc8c2_d192_hyperpartisan_100ep | 0.5575 | 0.9859 | 4.394 | 1814.0 | -0.6838 |
| twr_backbone_hyperpartisan_lean | 0.5275 | 1.8065 | 4.380 | 2635.7 | 0.8799 |

## longbench

| run | acc | loss | depth | throughput | depth corr |
| --- | ---: | ---: | ---: | ---: | ---: |
| twr_gmc8c2_d960_longbench_trec_100ep | 0.3600 | 3.2141 | 6.315 | 283.1 | 0.0000 |
| twr_gmc8c2_d576_longbench_trec_100ep | 0.3400 | 3.5201 | 6.213 | 599.6 | 0.0000 |
| twr_gmc8c2_d320_longbench_trec_100ep | 0.2800 | 3.2659 | 6.113 | 1136.8 | 0.0000 |
| twr_gmc8c2_d192_longbench_trec_100ep | 0.2600 | 3.4483 | 5.799 | 1205.9 | 0.0000 |
| twr_backbone_longbench_trec_lean | 0.1800 | 3.3970 | 5.860 | 1311.4 | 0.0000 |
| twr_backbone_longbench_lsht_lean | 0.1000 | 3.3430 | 6.189 | 1771.7 | 0.0000 |

## lra

| run | acc | loss | depth | throughput | depth corr |
| --- | ---: | ---: | ---: | ---: | ---: |
| twr_scale11p2m_gm8_lra_listops_100ep | 0.3398 | 2.0141 | 6.834 | 661.8 | 0.6994 |
| twr_scale17p4m_gm8_lra_listops_100ep | 0.3320 | 2.0579 | 6.929 | 620.3 | 0.0476 |
| twr_scale25m_gm8_lra_listops_100ep | 0.3281 | 2.0337 | 6.911 | 325.4 | 0.6430 |
| twr_scale6p3m_gm8_lra_listops_100ep | 0.3242 | 2.2296 | 6.896 | 1382.6 | -0.3060 |
| twr_scale11p2m_lra_listops_100ep | 0.3203 | 2.0194 | 6.790 | 448.6 | 0.1748 |
| twr_scale17p4m_lra_listops_100ep | 0.3203 | 2.1863 | 6.971 | 294.7 | 0.4219 |
| twr_scale6p3m_lra_listops_100ep | 0.3184 | 2.0272 | 6.781 | 736.0 | -0.5982 |
| twr_scale2p8m_lra_listops_100ep | 0.3164 | 2.3393 | 6.682 | 1415.6 | -0.3480 |
| twr_gmc8c2_d576_lra_listops_100ep | 0.3164 | 2.5393 | 4.618 | 1643.5 | 0.9282 |
| twr_gmc8c2_d960_lra_listops_100ep | 0.3125 | 2.1203 | 4.819 | 724.2 | 0.4468 |
| twr_gmc8c2_d320_lra_listops_100ep | 0.3047 | 2.1542 | 4.721 | 3971.2 | 0.2193 |
| twr_scale2p8m_gm8_lra_listops_100ep | 0.3047 | 2.1807 | 6.783 | 1928.7 | -0.7795 |
| twr_gmc8c2_d192_lra_listops_100ep | 0.3047 | 2.4766 | 4.480 | 7160.1 | 0.9129 |
| twr_scale57m_lra_listops | 0.3027 | 2.0301 | 6.984 | 145.3 | 0.1766 |
| twr_scale25m_lra_listops | 0.3027 | 2.0648 | 6.998 | 210.6 | 0.2021 |
| twr_scale25m_lra_listops_100ep | 0.3027 | 2.0950 | 6.953 | 210.5 | 0.0075 |
| twr_backbone_lra_listops_lean | 0.2520 | 2.0813 | 4.697 | 6061.6 | -0.8617 |
| twr_backbone_lra_listops_lean_fast | 0.2441 | 2.0377 | 3.258 | 9294.3 | -0.8248 |
| twr_backbone_lra_listops_lean_queries2 | 0.2305 | 2.0545 | 4.720 | 10302.8 | 0.8170 |
| twr_backbone_lra_listops_lean_think6 | 0.2109 | 2.1092 | 6.700 | 7755.7 | -0.8551 |
| twr_backbone_lra_listops_lean_think2 | 0.2109 | 2.1777 | 2.748 | 15373.0 | -0.8145 |
| twr_backbone_lra_listops | 0.2051 | 2.2067 | 2.000 | 31175.2 | 0.0000 |
| twr_backbone_lra_listops_lean_stride1 | 0.1816 | 2.1913 | 4.557 | 10467.6 | -0.8488 |
| twr_backbone_lra_listops_lean_window6 | 0.1777 | 2.2604 | 4.607 | 10488.1 | 0.9027 |

## parity_baselines

| run | acc | loss | depth | throughput | depth corr |
| --- | ---: | ---: | ---: | ---: | ---: |
| twr_debug | 0.9023 | 0.4067 | 1.621 | 13769.1 | 0.8691 |

## ruler

| run | acc | loss | depth | throughput | depth corr |
| --- | ---: | ---: | ---: | ---: | ---: |
| twr_scale11p2m_ruler_needle_100ep | 0.1074 | 2.8764 | 6.657 | 224.7 | -0.5879 |
| twr_gmc8c2_d192_ruler_needle_100ep | 0.1035 | 2.9631 | 6.418 | 3050.2 | -0.1947 |
| twr_scale6p3m_gm8_ruler_needle_100ep | 0.1035 | 3.7090 | 6.156 | 731.5 | -0.6792 |
| twr_scale2p8m_gm8_ruler_needle_100ep | 0.0996 | 3.6319 | 6.426 | 1034.9 | -0.7705 |
| twr_scale2p8m_ruler_needle_100ep | 0.0977 | 2.8247 | 6.478 | 724.7 | -0.5937 |
| twr_backbone_ruler_needle_lean_stride1 | 0.0977 | 2.9044 | 5.719 | 5945.9 | -0.5843 |
| twr_backbone_ruler_needle_lean | 0.0957 | 2.7914 | 6.237 | 6664.2 | -0.4394 |
| twr_gmc8c2_d576_ruler_needle_100ep | 0.0957 | 3.6355 | 6.293 | 706.6 | -0.4428 |
| twr_scale6p3m_ruler_needle_100ep | 0.0957 | 4.2424 | 5.398 | 370.0 | -0.9379 |
| twr_backbone_ruler_needle_lean_think8 | 0.0938 | 2.9884 | 7.255 | 6295.9 | -0.3576 |
| twr_gmc8c2_d960_ruler_needle_100ep | 0.0938 | 4.2855 | 6.058 | 313.2 | -0.7249 |
| twr_backbone_ruler_needle_lean_think4 | 0.0918 | 2.7898 | 4.410 | 10438.7 | 0.0587 |
| twr_gmc8c2_d320_ruler_needle_100ep | 0.0918 | 2.8896 | 6.457 | 1665.3 | -0.1815 |
| twr_scale17p4m_ruler_needle_100ep | 0.0879 | 4.2995 | 6.107 | 148.7 | -0.9374 |
| twr_backbone_ruler_needle_lean_256 | 0.0859 | 2.7849 | 6.125 | 7776.8 | 0.0000 |
| twr_backbone_ruler_needle_lean_fast | 0.0840 | 2.7808 | 5.111 | 7005.5 | -0.3342 |
| twr_backbone_ruler_needle_lean_1024 | 0.0840 | 2.7982 | 6.136 | 3926.3 | 0.2735 |
| twr_backbone_ruler_needle_lean_window10 | 0.0820 | 2.7763 | 6.147 | 7821.1 | 0.8092 |
| twr_backbone_ruler_needle_lean_queries4 | 0.0762 | 2.8209 | 6.131 | 7819.8 | 0.3578 |
| twr_backbone_ruler_needle_lean_memory | 0.0703 | 2.8055 | 6.862 | 5690.6 | 0.2474 |
