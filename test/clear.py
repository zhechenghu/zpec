import os
import shutil

result_path = os.path.join(os.path.dirname(__file__), "result")
shutil.rmtree(result_path)
# create a new directory
os.mkdir(result_path)
