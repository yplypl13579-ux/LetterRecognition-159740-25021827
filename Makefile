# Run from a Qt/MinGW command prompt with qmake and g++ on PATH.
# Build output stays in build/; qmake never overwrites this wrapper Makefile.
.PHONY: all gui experiments test
all: gui
gui:
	@if not exist build mkdir build
	qmake -o build/Makefile LetterRecognition.pro
	$(MAKE) -C build
experiments:
	@if not exist build mkdir build
	g++ -std=c++17 -O3 -Wall -Wextra backpropagation.cpp experiments.cpp -o build/experiments.exe
test: experiments
	build\experiments.exe --self-test build/test_scratch
