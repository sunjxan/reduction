# echo "cpu:"
# nvcc -O2 -std=c++17 -Xcompiler -Wall cpu_1.cu -o a && ./a
# nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP cpu_1.cu -o a && ./a
# nvcc -O2 -std=c++17 -Xcompiler -Wall cpu_2.cu -o a && ./a
# nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP cpu_2.cu -o a && ./a
# nvcc -O2 -std=c++17 -Xcompiler -Wall cpu_3.cu -o a && ./a
# nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP cpu_3.cu -o a && ./a
# nvcc -O2 -std=c++17 -Xcompiler -Wall cpu_4.cu -o a && ./a
# nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP cpu_4.cu -o a && ./a
# nvcc -O2 -std=c++17 -Xcompiler -Wall cpu_5.cu -o a && ./a
# nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP cpu_5.cu -o a && ./a
# nvcc -O2 -std=c++17 -Xcompiler -Wall cpu_6.cu -o a && ./a
# nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP cpu_6.cu -o a && ./a
# echo ""
echo "cuda:"
# nvcc -O2 -std=c++17 -Xcompiler -Wall cuda_1.cu -o a && ./a
# nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP cuda_1.cu -o a && ./a
# nvcc -O2 -std=c++17 -Xcompiler -Wall cuda_2.cu -o a && ./a
# nvcc -O2 -std=c++17 -arch=sm_60 -Xcompiler -Wall -DUSE_DP cuda_2.cu -o a && ./a
# nvcc -O2 -std=c++17 -Xcompiler -Wall cuda_3.cu -o a && ./a
# nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP cuda_3.cu -o a && ./a
nvcc -O2 -std=c++17 -Xcompiler -Wall cuda_4.cu -o a && ./a
nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP cuda_4.cu -o a && ./a
nvcc -O2 -std=c++17 -Xcompiler -Wall cuda_5.cu -o a && ./a
nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP cuda_5.cu -o a && ./a
nvcc -O2 -std=c++17 -Xcompiler -Wall cuda_6.cu -o a && ./a
nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP cuda_6.cu -o a && ./a
nvcc -O2 -std=c++17 -Xcompiler -Wall cuda_7.cu -o a && ./a
nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP cuda_7.cu -o a && ./a
echo ""
echo "optimize:"
nvcc -O2 -std=c++17 -Xcompiler -Wall optimize_1.cu -o a && ./a
nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP optimize_1.cu -o a && ./a
nvcc -O2 -std=c++17 -Xcompiler -Wall optimize_2.cu -o a && ./a
nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP optimize_2.cu -o a && ./a
nvcc -O2 -std=c++17 -Xcompiler -Wall optimize_3.cu -o a && ./a
nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP optimize_3.cu -o a && ./a
nvcc -O2 -std=c++17 -Xcompiler -Wall optimize_4.cu -o a && ./a
nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP optimize_4.cu -o a && ./a
nvcc -O2 -std=c++17 -Xcompiler -Wall optimize_5.cu -o a && ./a
nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP optimize_5.cu -o a && ./a
nvcc -O2 -std=c++17 -Xcompiler -Wall optimize_6.cu -o a && ./a
nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP optimize_6.cu -o a && ./a
nvcc -O2 -std=c++17 -Xcompiler -Wall optimize_7.cu -o a && ./a
nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP optimize_7.cu -o a && ./a
nvcc -O2 -std=c++17 -Xcompiler -Wall optimize_8.cu -o a && ./a
nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP optimize_8.cu -o a && ./a
nvcc -O2 -std=c++17 -Xcompiler -Wall optimize_9.cu -o a && ./a
nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP optimize_9.cu -o a && ./a