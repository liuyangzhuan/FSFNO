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

cd ../examples/NIMROD2D/
cd  plots_Colrs
mkdir eror nrml pred trut MxNm
echo "pwd= $(pwd)"

cp */*eror.png eror/
cp */*nrml.png nrml/
cp */*pred.png pred/
cp */*trut.png trut/
cp */*MxNm.png MxNm/

cd eror
convert -delay 200 -loop 0 *.png anim_eror.gif
echo "pwd= $(pwd)"

cd ../pred
convert -delay 200 -loop 0 *.png anim_pred.gif
echo "pwd= $(pwd)"

cd ../trut
convert -delay 200 -loop 0 *.png anim_trut.gif
echo "pwd= $(pwd)"

cd ../nrml
convert -delay 200 -loop 0 *.png anim_nrml.gif
echo "pwd= $(pwd)"

cd ../MxNm
convert -delay 200 -loop 0 *.png anim_MxNm.gif
echo "pwd= $(pwd)"

cd ../

echo "pwd= $(pwd)"

#! /bin/bash

# ar = $(identify -format '%w '  *.png)
# max=${ar[0]}
# for n in "${ar[@]}" ; do
#     ((n > max)) && max=$n
# done

cd ../
echo "pwd= $(pwd)"

#/bin/bash sameSizes.sh
./sameSizes.sh
# source sameSizes.sh