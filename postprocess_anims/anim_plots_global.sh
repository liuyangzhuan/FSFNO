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

# cd  plots_Colrs
# echo "02"
# rm -r pdf
# echo "04"
# cd ../examples/NIMROD2D/

my_folder_array=(plots_pmesh_RdBu plots_pmesh plots_Colrs)

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
    echo "21 anim_plots_global.sh my_folder_array_iterator: $my_folder_array_iterator"
    cd  "$my_folder_array_iterator";
    echo "02"
    rm -r pdf
    echo "04"

    for dir_globals in */ ; do
        echo "045"
        cd "$dir_globals";
        echo "anim_plots_global.sh 30 pwd= $(pwd)"
        rm -r pdf
        echo "055"
        for dir_field in */ ; do
            echo "06"
            cd "$dir_field";
            echo "anim_plots_global.sh 36 pwd= $(pwd)"
            rm -r pdf
            echo "08"
            for dir_eq in */ ; do
                echo "10"
                cd "$dir_eq";
                echo "anim_plots_global.sh 42 pwd= $(pwd)"
                echo "12"
                rm -r pdf
                for dir_vec in */ ; do
                    echo "14"
                    cd "$dir_vec";
                    echo "anim_plots_global.sh 48 pwd= $(pwd)"
                    echo "$dir_vec"
                    echo "16"
                    
                    mkdir eror nrml pred trut MxNm
                    echo "anim_plots_global.sh 53 pwd= $(pwd)"
                    echo "18"

                    cp */*eror.png eror/
                    cp */*nrml.png nrml/
                    cp */*pred.png pred/
                    cp */*trut.png trut/
                    cp */*MxNm.png MxNm/

                    echo "32"

                    cd eror
                    convert -delay 700 -loop 0 *.png anim_eror.gif
                    echo "anim_plots_global.sh 66 pwd= $(pwd)"

                    echo "38"

                    cd ../pred
                    convert -delay 700 -loop 0 *.png anim_pred.gif
                    echo "anim_plots_global.sh 72 pwd= $(pwd)"

                    echo "44"

                    cd ../trut
                    convert -delay 700 -loop 0 *.png anim_trut.gif
                    echo "anim_plots_global.sh 78 pwd= $(pwd)"

                    echo "48"

                    cd ../nrml
                    convert -delay 700 -loop 0 *.png anim_nrml.gif
                    echo "anim_plots_global.sh 84 pwd= $(pwd)"

                    echo "52"

                    cd ../MxNm
                    convert -delay 700 -loop 0 *.png anim_MxNm.gif
                    echo "anim_plots_global.sh 90 pwd= $(pwd)"


                    echo "62"
                    cd ../
                    echo "anim_plots_global.sh 62 pwd= $(pwd)"


                    echo "64"

                    cd ../
                    echo "anim_plots_global.sh 101 pwd= $(pwd)"

                done

                echo "68"
                cd ../
                echo "anim_plots_global.sh 107 pwd= $(pwd)"
            done

            echo "70"
            cd ../
            echo "anim_plots_global.sh 112 pwd= $(pwd)"
        done

        echo "74"
        cd ../
        echo "anim_plots_global.sh 117  pwd= $(pwd)"
    done

    #! /bin/bash

    # ar = $(identify -format '%w '  *.png)
    # max=${ar[0]}
    # for n in "${ar[@]}" ; do
    #     ((n > max)) && max=$n
    # done

    cd ../
    echo "anim_plots_global.sh 129  pwd= $(pwd)"

done

#/bin/bash sameSizes.sh
echo "anim_plots_global.sh 134  pwd= $(pwd)"
./sameSizes_global.sh
# source sameSizes.sh
echo "anim_plots_global.sh 137  pwd= $(pwd)"

echo "anim_plots_global.sh 139  pwd= $(pwd)"
./anim_cp_dir_glob.sh
echo "anim_plots_global.sh 141  pwd= $(pwd)"

























