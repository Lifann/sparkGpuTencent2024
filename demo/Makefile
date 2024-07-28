all: libyour_api.so a.out

libyour_api.so: your_api.cu cuda_fns.cu
	nvcc -Xcompiler -fPIC -shared -std=c++17 -arch=sm_70 $^ -o $@

a.out: caller.cu
	nvcc -std=c++17 -arch=sm_70 -L/data/compile_demo $^ -o $@ -lyour_api

clean:
	rm libyour_api.so
	rm a.out
