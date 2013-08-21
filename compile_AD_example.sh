#!/bin/bash
#Tested on Ubuntu 13.04 64bit
#Compiled with GCC 4.7
#Armadillo API version 3.91
#Trilinos API 11.0.3 configured with Teuchos and Sacado packages enabled

g++ automatic_differentiation.cpp -larmadillo -lteuchos -o adexample.exe
