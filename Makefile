CPPFLAGS=-std=c++14 -Wall -pedantic -Wno-long-long -O2 -I /usr/local/Cellar/eigen/3.3.9/include/eigen3

run: m
	./m

m: main.o
	$(CXX) $(CPPFLAGS) main.o -o m

clean:
	- rm m *.o
