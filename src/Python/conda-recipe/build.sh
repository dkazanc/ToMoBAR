mkdir ${SRC_DIR}/TomoRec
cp -r "${RECIPE_DIR}/../../../" ${SRC_DIR}/TomoRec

cd ${SRC_DIR}/TomoRec/src/Python
$PYTHON setup.py install
