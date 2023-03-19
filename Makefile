<<<<<<< HEAD
default: build

build: source/*.cpp
	g++ -O3 -Wall -std=c++23 $^ -o NeuralNetwork

run: build
	./NeuralNetwork

clean:
=======
default: build

build: source/*.cpp
	g++ -O3 -Wall -std=c++23 $^ -o NeuralNetwork

run: build
	./NeuralNetwork

clean:
>>>>>>> 5ecd6b69e30d071fc66c8798299c977f0b62598d
	rm NeuralNetwork