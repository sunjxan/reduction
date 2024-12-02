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

# nvcc -O2 -std=c++17 -Xcompiler -Wall cuda_1.cu -o a && ./a
# nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP cuda_1.cu -o a && ./a
# nvcc -O2 -std=c++17 -Xcompiler -Wall cuda_2.cu -o a && ./a
# nvcc -O2 -std=c++17 -arch=sm_60 -Xcompiler -Wall -DUSE_DP cuda_2.cu -o a && ./a
nvcc -O2 -std=c++17 -Xcompiler -Wall cuda_3.cu -o a && ./a
nvcc -O2 -std=c++17 -arch=sm_60 -Xcompiler -Wall -DUSE_DP cuda_3.cu -o a && ./a