@ECHO off
WHERE pip 
IF %ERRORLEVEL% NEQ 0 (ECHO curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py 
   python get-pip.py
 ) ELSE (ECHO pip found)
 


IF exist env ( echo env exists ) ELSE ( pip install virtualenv
virtualenv env)


call env\Scripts\activate.bat

echo installing requirements...
pip install -r req.txt

cd code/
echo running code..
echo press q to stop
python main.py -s webcam


PAUSE






