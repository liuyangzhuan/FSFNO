#! /bin/bash

# FSFNO Copyright (c) 2024, The Regents of the University of California,
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

# cd  plots_Colrs
# echo "02_2"
# rm -r pdf
# echo "04_2"

cd ../examples/NIMROD2D/

echo "Here Start of sanim_cp_dir_glob.sh $(pwd)"
echo "pwd= $(pwd)"
pwd_orig=$pwd
# cd  plots_Colrs
# echo "pwd= $(pwd)"
# echo "04"
# rm -r pdf

my_folder_array=(plots_pmesh_RdBu plots_Colrs plots_pmesh)

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
    echo "21 anim_cp_dir_glob.sh.sh my_folder_array_iterator: $my_folder_array_iterator"
    cd  "$my_folder_array_iterator";
    echo "02_2"
    rm -r pdf
    echo "04_2"

    for dir_globals in */ ; do
        echo "045_2"
        cd "$dir_globals";
        echo "anim_cp_dir_glob.sh 40 pwd= $(pwd)"
        rm -r pdf
        echo "055_2"
        for dir_field in */ ; do
            echo "06_2"
            cd "$dir_field";
            echo "anim_cp_dir_glob.sh 46 pwd= $(pwd)"
            rm -r pdf
            echo "08_2"
            for dir_eq in */ ; do
                echo "10_2"
                cd "$dir_eq";
                echo "anim_cp_dir_glob.sh 50 pwd= $(pwd)"
                echo "12"
                rm -r pdf
                for dir_vec in */ ; do
                    echo "14_2"
                    cd "$dir_vec";
                    echo "anim_cp_dir_glob.sh 58 pwd= $(pwd)"
                    echo "$dir_vec"
                    echo "16_2"
                    
                    mkdir anim
                    echo "anim_cp_dir_glob.sh 63 pwd= $(pwd)"
                    echo "18"

                    cp eror/anim_sameSize.gif anim/
                    mv anim/anim_sameSize.gif anim/anim_sameSize_eror.gif

                    cp nrml/anim_sameSize.gif anim/
                    mv anim/anim_sameSize.gif anim/anim_sameSize_nrml.gif

                    cp pred/anim_sameSize.gif anim/
                    mv anim/anim_sameSize.gif anim/anim_sameSize_pred.gif

                    cp trut/anim_sameSize.gif anim/
                    mv anim/anim_sameSize.gif anim/anim_sameSize_trut.gif

                    cp MxNm/anim_sameSize.gif anim/
                    mv anim/anim_sameSize.gif anim/anim_sameSize_MxNm.gif


                    echo "32_2"


                    echo "62_2"
                    echo "anim_cp_dir_glob.sh 86 pwd= $(pwd)"


                    echo "64_2"

                    cd ../
                    echo "anim_cp_dir_glob.sh 92 pwd= $(pwd)"

                done

                echo "68_2"
                cd ../
                echo "anim_cp_dir_glob.sh 98 pwd= $(pwd)"
            done

            echo "70"_2
            cd ../
            echo "anim_cp_dir_glob.sh 103 pwd= $(pwd)"
        done

        echo "74_2"
        cd ../
        echo "anim_cp_dir_glob.sh 108 pwd= $(pwd)"
    done

    #! /bin/bash

    # ar = $(identify -format '%w '  *.png)
    # max=${ar[0]}
    # for n in "${ar[@]}" ; do
    #     ((n > max)) && max=$n
    # done

    echo "anim_cp_dir_glob.sh 119 pwd= $(pwd)"
    cd ../
    echo "anim_cp_dir_glob.sh 121 pwd= $(pwd)"
done

echo "anim_cp_dir_glob.sh 124 pwd= $(pwd)"