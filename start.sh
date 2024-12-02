nvcc -O2 -std=c++17 -Xcompiler -Wall cpu_1.cu -o a && ./a
nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP cpu_1.cu -o a && ./a
nvcc -O2 -std=c++17 -Xcompiler -Wall cpu_2.cu -o a && ./a
nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP cpu_2.cu -o a && ./a
nvcc -O2 -std=c++17 -Xcompiler -Wall cpu_3.cu -o a && ./a
nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP cpu_3.cu -o a && ./a
nvcc -O2 -std=c++17 -Xcompiler -Wall cpu_4.cu -o a && ./a
nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP cpu_4.cu -o a && ./a