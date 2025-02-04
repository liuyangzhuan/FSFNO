#! /bin/bash

# Sparsified Time-dependent PDEs FNO (STFNO) Copyright (c) 2025, The Regents of 
# the University of California, through Lawrence Berkeley National Laboratory 
# (subject to receipt of any required approvals from the U.S.Dept. of Energy).  
# All rights reserved.
#
# If you have questions about your rights to use or distribute this software,
# please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.
#
# NOTICE. This Software was developed under funding from the U.S. Department
# of Energy and the U.S. Government consequently retains certain rights.
# As such, the U.S. Government has been granted for itself and others acting
# on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in
# the Software to reproduce, distribute copies to the public, prepare
# derivative works, and perform publicly and display publicly, and to permit
# other to do so.


# ar = $(identify -format '%w '  *.png)
# max=${ar[0]}
# for n in "${ar[@]}" ; do
#     ((n > max)) && max=$n
# done
cd ../examples/NIMROD2D/

pwd_orig=$pwd
for d in ./*/*/ ; do 
  cd "$d" ;
  echo "pwd= $(pwd)"
  max_w=0
  for n in $(identify -format '%w '  *.png); do
    ((n > max_w)) && max_w=$n
    #echo "n=$n , max_w=$max_w"
  done
  echo "max_width_px=$max_w"

  max_h=0
  for n in $(identify -format '%h '  *.png); do
    ((n > max_h)) && max_h=$n
    #echo "n=$n , max_w=$max_h"
  done
  #echo $max_h
  echo "max_height_px=$max_h"

  arg_3_beg="$max_w"x$"$max_h"
  arg3="-resize","$arg_3_beg"
  #$arg3 = '-resize', '1200x1200'

  #$arg4='-size', '$arg_3_beg'

  mkdir -p CONVERTED
  for i in *.png; do 
      convert $i -resize $arg_3_beg -size $arg_3_beg xc:white +swap -gravity center -composite ./CONVERTED/$i; 
  done
  convert -delay 100 -dispose previous -loop 0 ./CONVERTED/*.png animated.gif
  rm -rf CONVERTED
  echo "pwd= $(pwd)"
  cd ../..
done
