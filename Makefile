default: build

build: source/*.cpp
	g++ -O3 -Wall -std=c++23 $^ -o NeuralNetwork

run: build
	./NeuralNetwork

clean:
	rm NeuralNetwork