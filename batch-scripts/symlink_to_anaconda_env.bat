set ANACONDA_ENV=data-analysis
set ANACONDA_ROOT=%userprofile%\local\anaconda3
mklink %ANACONDA_ROOT%\envs\%ANACONDA_ENV%\lib\site-packages\odmr_lib.py %CD%\odmr_lib.py
pause