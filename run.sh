#!/bin/bash

rm -rf CMakeCache.txt cmake_install.cmake CMakeFiles Makefile
cp gen_so.txt CMakeLists.txt
cmake .
make
mv ./libkeeppose.so ./lib/

rm -rf main
rm -rf CMakeCache.txt cmake_install.cmake CMakeFiles Makefile
cp gen_exe.txt CMakeLists.txt
cmake .
make

./main

