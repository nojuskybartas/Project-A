# Project-A
 
Note: you need python < 3.10 for this to work - some of the dependencies are not compiled for python 3.10 yet.
Built on python 3.9.

You need to install the requirements with:
```
pip install -r requirements.txt
```

If pip is not found, you need to add the scripts folder of the python installation you're using to the system
environment variables.

Note this work is very rough, there is a lot to improve.

# Summary

the main function is under model/main.py

at the bottom of main.py you can choose whether you want to train, or test a trained model. 

When training a model you will be asked to describe the model. This will be the name of the model, which you can then use to access the same model/weights when trying to test it.
