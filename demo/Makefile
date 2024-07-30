all: libyour_api.so verify

#libyour_api.so: your_api.cu cuda_fns.cu
#	nvcc -Xcompiler -fPIC -shared -std=c++17 -arch=sm_70 $^ -o $@

verify: verify.cu
	nvcc -std=c++17 -O2 -G -g -lineinfo -arch=sm_70 -L/root/Gen $^ -o $@ -lyour_api

clean:
	#rm libyour_api.so
	rm verify
