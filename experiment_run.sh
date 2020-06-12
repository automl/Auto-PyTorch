#!/bin/bash

file=$1
export dir="$2"

if ! [ -e "$file" ] ; then     # spaces inside square brackets
    echo "$0: $file does not exist" >&2  # error message includes $0 and goes to stderr
    exit 1                   # exit code is non-zero for error
fi


NUMBERS=$(<$file)
for NUM in $NUMBERS
do
   export task=$(echo $NUM)     
   msub -V -t 1-16 investigating_dl.moab
done

