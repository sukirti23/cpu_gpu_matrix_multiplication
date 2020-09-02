all: main.cpp
#	g++ -O3 -std=c++11 main.cpp -o matrix_mul -lOpenCL
	g++ -O3 -std=c++11 main.cpp -o matrix_mul -lOpenCL

clean:
	rm -f matrix_mul
