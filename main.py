from model_testing import predict
from ui import choosefile

#main
path = choosefile()
predict(path)