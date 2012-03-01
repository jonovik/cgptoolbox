import pickle
from cgp.physmod.cellmlmodel import Cellmlmodel
from cgp.virtexp.elphys.paceable import Paceable

class Model(Cellmlmodel, Paceable):
    pass

m = Model()

print pickle.dumps(m)

print m