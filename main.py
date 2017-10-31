from questionVectors import execute
from classifier import runClassifer

data = execute("googleNews", "sum")
runClassifer(data)
