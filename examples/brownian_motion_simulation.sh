#!/bin/bash
#
cp brownian_motion_simulation.h /$HOME/include
#
gcc -c -I/$HOME/include brownian_motion_simulation.c
if [ $? -ne 0 ]; then
  echo "Errors compiling brownian_motion_simulation.c"
  exit
fi
#
mv brownian_motion_simulation.o ~/libc/$ARCH/brownian_motion_simulation.o
#
echo "Library installed as ~/libc/$ARCH/brownian_motion_simulation.o"
