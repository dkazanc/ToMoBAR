mkdir ${SRC_DIR}/ToMoBAR
cp -r "${RECIPE_DIR}/../../../" ${SRC_DIR}/ToMoBAR

cd ${SRC_DIR}/ToMoBAR/src/Python
$PYTHON setup.py install
