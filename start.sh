# echo "cpu:"
# nvcc -O2 -std=c++17 -Xcompiler -Wall cpu_1.cu -o cpu_1.out && ./cpu_1.out
# nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP cpu_1.cu -o cpu_1_dp.out && ./cpu_1_dp.out
# nvcc -O2 -std=c++17 -Xcompiler -Wall cpu_2.cu -o cpu_2.out && ./cpu_2.out
# nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP cpu_2.cu -o cpu_2_dp.out && ./cpu_2_dp.out
# nvcc -O2 -std=c++17 -Xcompiler -Wall cpu_3.cu -o cpu_3.out && ./cpu_3.out
# nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP cpu_3.cu -o cpu_3_dp.out && ./cpu_3_dp.out
# nvcc -O2 -std=c++17 -Xcompiler -Wall cpu_4.cu -o cpu_4.out && ./cpu_4.out
# nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP cpu_4.cu -o cpu_4_dp.out && ./cpu_4_dp.out
# nvcc -O2 -std=c++17 -Xcompiler -Wall cpu_5.cu -o cpu_5.out && ./cpu_5.out
# nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP cpu_5.cu -o cpu_5_dp.out && ./cpu_5_dp.out
# nvcc -O2 -std=c++17 -Xcompiler -Wall cpu_6.cu -o cpu_6.out && ./cpu_6.out
# nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP cpu_6.cu -o cpu_6_dp.out && ./cpu_6_dp.out
# echo ""
echo "cuda:"
# nvcc -O2 -std=c++17 -Xcompiler -Wall cuda_1.cu -o cuda_1.out && ./cuda_1.out
# nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP cuda_1.cu -o cuda_1_dp.out && ./cuda_1_dp.out
# nvcc -O2 -std=c++17 -Xcompiler -Wall cuda_2.cu -o cuda_2.out && ./cuda_2.out
# nvcc -O2 -std=c++17 -arch=sm_60 -Xcompiler -Wall -DUSE_DP cuda_2.cu -o cuda_2_dp.out && ./cuda_2_dp.out
# nvcc -O2 -std=c++17 -Xcompiler -Wall cuda_3.cu -o cuda_3.out && ./cuda_3.out
# nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP cuda_3.cu -o cuda_3_dp.out && ./cuda_3_dp.out
nvcc -O2 -std=c++17 -Xcompiler -Wall cuda_4.cu -o cuda_4.out && ./cuda_4.out
nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP cuda_4.cu -o cuda_4_dp.out && ./cuda_4_dp.out
nvcc -O2 -std=c++17 -Xcompiler -Wall cuda_5.cu -o cuda_5.out && ./cuda_5.out
nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP cuda_5.cu -o cuda_5_dp.out && ./cuda_5_dp.out
nvcc -O2 -std=c++17 -Xcompiler -Wall cuda_6.cu -o cuda_6.out && ./cuda_6.out
nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP cuda_6.cu -o cuda_6_dp.out && ./cuda_6_dp.out
nvcc -O2 -std=c++17 -Xcompiler -Wall cuda_7.cu -o cuda_7.out && ./cuda_7.out
nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP cuda_7.cu -o cuda_7_dp.out && ./cuda_7_dp.out
echo ""
echo "optimize:"
nvcc -O2 -std=c++17 -Xcompiler -Wall optimize_1.cu -o optimize_1.out && ./optimize_1.out
nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP optimize_1.cu -o optimize_1_dp.out && ./optimize_1_dp.out
nvcc -O2 -std=c++17 -Xcompiler -Wall optimize_2.cu -o optimize_2.out && ./optimize_2.out
nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP optimize_2.cu -o optimize_2_dp.out && ./optimize_2_dp.out
nvcc -O2 -std=c++17 -Xcompiler -Wall optimize_3.cu -o optimize_3.out && ./optimize_3.out
nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP optimize_3.cu -o optimize_3_dp.out && ./optimize_3_dp.out
nvcc -O2 -std=c++17 -Xcompiler -Wall optimize_4.cu -o optimize_4.out && ./optimize_4.out
nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP optimize_4.cu -o optimize_4_dp.out && ./optimize_4_dp.out
nvcc -O2 -std=c++17 -Xcompiler -Wall optimize_5.cu -o optimize_5.out && ./optimize_5.out
nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP optimize_5.cu -o optimize_5_dp.out && ./optimize_5_dp.out
nvcc -O2 -std=c++17 -Xcompiler -Wall optimize_6.cu -o optimize_6.out && ./optimize_6.out
nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP optimize_6.cu -o optimize_6_dp.out && ./optimize_6_dp.out
nvcc -O2 -std=c++17 -Xcompiler -Wall optimize_7.cu -o optimize_7.out && ./optimize_7.out
nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP optimize_7.cu -o optimize_7_dp.out && ./optimize_7_dp.out
nvcc -O2 -std=c++17 -Xcompiler -Wall optimize_8.cu -o optimize_8.out && ./optimize_8.out
nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP optimize_8.cu -o optimize_8_dp.out && ./optimize_8_dp.out
nvcc -O2 -std=c++17 -Xcompiler -Wall optimize_9.cu -o optimize_9.out && ./optimize_9.out
nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP optimize_9.cu -o optimize_9_dp.out && ./optimize_9_dp.out
