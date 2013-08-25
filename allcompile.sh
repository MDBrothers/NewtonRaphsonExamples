#!/bin/bash
#Tested on Ubuntu 13.04 64bit
#Compiled with GCC 4.7
#Trilinos API 11.0.3 configured with Teuchos and Sacado packages enabled

g++ automatic_differentiation.cpp -lteuchos -lblas -llapack -o adexample.exe

g++ forward_difference.cpp -lteuchos -lblas -llapack -o fdexample.exe

g++ complex_step.cpp -lteuchos -lblas -llapack -o csexample.exe
