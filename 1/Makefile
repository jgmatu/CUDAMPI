all:
	nvcc -arch=sm_20 -gencode arch=compute_20,code=sm_20 -g -G hello.cu -o hello
	nvcc -arch=sm_20 -gencode arch=compute_20,code=sm_20 -g -G suma.cu -o suma
	nvcc -arch=sm_20 -gencode arch=compute_20,code=sm_20 -g -G suma_afin.cu -o suma_afin
