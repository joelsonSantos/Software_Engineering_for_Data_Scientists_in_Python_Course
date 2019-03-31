# package used to check the style (identation) of the code
import pycodestyle

check_code = pycodestyle.StyleGuide()
result = check_code.check_files(['code_test.py']) # can use much more files to check. 
print(result.messages)