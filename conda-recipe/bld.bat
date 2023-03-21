IF NOT DEFINED VERSION (
ECHO VERSION Not Defined.
exit 1
)

"%PYTHON%" -m pip install .
if errorlevel 1 exit 1