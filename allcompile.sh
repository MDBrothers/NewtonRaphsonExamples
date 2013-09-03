#!/bin/bash
#Tested on Ubuntu 13.04 64bit
#Compiled with GCC 4.7
#Trilinos API 11.0.3 configured with Teuchos and Sacado packages enabled

g++ NRexamples.cpp -lteuchos -lblas -llapack -o NRexamples.exe

g++ NRComparison.cpp -lteuchos -lblas -llapack -o NRComparison.exe

