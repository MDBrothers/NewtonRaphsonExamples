#!/bin/bash
#Tested on Ubuntu 13.04 64bit
#Compiled with GCC 4.7
#Armadillo API version 3.91
g++ -g forward_difference.cpp -larmadillo -lteuchos -lblas -llapack -o fdexample.exe
