Cdeb:
	g++ -g -Isource/ source/test_container_cpu.cpp -o test_container_cpu.bin
Crel:
	g++ -O2 -Isource/ source/test_container_cpu.cpp -o test_container_cpu.bin
Gdeb:
	nvcc -g -Isource/ source/test_container_gpu.cpp -o test_container_gpu.bin
Grel:
	nvcc -O2 -Isource/ source/test_container_gpu.cpp -o test_container_gpu.bin
