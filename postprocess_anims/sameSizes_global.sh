#! /bin/bash

# STFNO Copyright (c) 2024, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of any
# required approvals from the U.S.Dept. of Energy) and the University of
# California, Berkeley.  All rights reserved.
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

echo "Here Start of sameSize_global.sh $(pwd)"
echo "sameSizes_global.sh 11 pwd= $(pwd)"
pwd_orig=$pwd
# cd  plots_Colrs
# echo "pwd= $(pwd)"
# echo "04"
# rm -r pdf

my_folder_array=(plots_Colrs plots_pmesh plots_pmesh_RdBu)

# Access elements
echo "First folder element: ${my_folder_array[0]}"
echo "Second folder element: ${my_folder_array[1]}"
echo "All folder  elements: ${my_folder_array[@]}"
# Add elements
# my_array+=(orange)
# echo "After adding an element: ${my_array[@]}"

# Length of array
echo "Length of folder array: ${#my_folder_array[@]}"

# Iterate over elements
for my_folder_array_iterator in "${my_folder_array[@]}"; do
    echo "33 sameSizes_global.sh my_folder_array_iterator: $my_folder_array_iterator"
    cd  "$my_folder_array_iterator";
    echo "02"
    rm -r pdf
    echo "04"

    #cd  plots_Colrs
    echo "sameSizes_global.sh 40 pwd= $(pwd)"
    echo "04"
    rm -r pdf

    for dir_global in */ ; do
      parent="$(pwd)"
      path=$parent/$dir_global
      echo $path

      #if [ -d "$path" ]; then
        echo "$path exists."
        cd "$dir_global";
        echo "sameSizes_global.sh 52 pwd= $(pwd)"
        echo "07"
        
        rm -r pdf
      for dir_field in */ ; do
        parent="$(pwd)"
        path=$parent/$dir_field
        echo $path
        echo "07_2"
        #if [ -d "$path" ]; then
          echo "$path exists."
          cd "$dir_field";
          echo "sameSizes_global.sh 64 pwd= $(pwd)"
          echo "07_3"
          
          rm -r pdf
          for dir_eq in */ ; do
            parent="$(pwd)"
            path=$parent/$dir_eq
            echo $path
            echo "08_3"

            #if [ -d "$path" ]; then
              echo "$path exists."
              cd "$dir_eq";
              echo "sameSizes_global.sh 77 pwd= $(pwd)"
              echo "09"

              rm -r pdf
              for dir_vec in */ ; do
                parent="$(pwd)"
                path=$parent/$dir_vec
                echo $path
                echo "09_3"

                #if [ -d "$path" ]; then
                  cd "$dir_vec";
                  echo "sameSizes_global.sh 89 pwd= $(pwd)"
                  echo "$dir_vec"
                  echo "12"

                  rm -r pdf
                  for dir_outstep in ./*/ ; do 
                    parent="$(pwd)"
                    path=$parent/$dir_outstep
                    echo $path

                    echo "12_3"

                    # if [ -d "$path" ]; then
                      cd "$dir_outstep" ;
                      echo "sameSizes_global.sh 103 pwd= $(pwd)"
                      echo "14"
                      max_w=0
                      for n in $(identify -format '%w '  *.png); do
                        ((n > max_w)) && max_w=$n
                        #echo "n=$n , max_w=$max_w"
                        echo "18"
                      done
                      echo "max_width_px=$max_w"
                      echo "20"

                      max_h=0
                      for n in $(identify -format '%h '  *.png); do
                        ((n > max_h)) && max_h=$n
                        #echo "n=$n , max_w=$max_h"
                        echo "22"
                      done
                      echo "24"
                      #echo $max_h
                      echo "max_height_px=$max_h"
                      echo "28"

                      arg_3_beg="$max_w"x$"$max_h"
                      arg3="-resize","$arg_3_beg"
                      #$arg3 = '-resize', '1200x1200'
                      echo "32"

                      #$arg4='-size', '$arg_3_beg'

                      mkdir -p CONVERTED
                      echo "34"
                      for i in *.png; do 
                          convert $i -resize $arg_3_beg -size $arg_3_beg xc:white +swap -gravity center -composite ./CONVERTED/$i; 
                          echo "38"
                      done
                      echo "40"
                      convert -delay 35 -dispose previous -loop 0 ./CONVERTED/*.png anim_sameSize.gif
                      echo "42"
                      rm -rf CONVERTED
                      echo "sameSizes_global.sh 142 pwd= $(pwd)"
                      echo "44"
                      cd ../

                      echo "48"
                    # fi
                  done
                  echo "50"

                  cd ../
                  echo "sameSizes_global.sh 152 pwd= $(pwd)"
                  echo "54"
                #fi
              done
              echo "56"
              cd ../
              echo "sameSizes_global.sh 158 pwd= $(pwd)"
              echo "58"
            #fi
          done
          echo "60"
          cd ../
          echo "sameSizes_global.sh 164 pwd= $(pwd)"
          echo "66"
        #fi

      done

      echo "68"
      cd ../
      echo "sameSizes_global.sh 172 pwd= $(pwd)"
      echo "70"
    done

    echo "sameSizes_global.sh 176 pwd= $(pwd)"

    cd ../

    echo "80"
done

echo "sameSizes_global.sh 181 pwd= $(pwd)"