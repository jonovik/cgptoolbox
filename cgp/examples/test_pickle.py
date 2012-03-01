import pickle
from cgp.physmod.cellmlmodel import Cellmlmodel
from cgp.virtexp.elphys.paceable import Paceable

class Model(Cellmlmodel, Paceable):
    pass

m = Model(exposure_workspace="c7f7ced1e002d9f0af1b56b15a873736/tentusscher_noble_noble_panfilov_2004_a",
        rename={"y": {"Na_i": "Nai", "Ca_i": "Cai", "K_i": "Ki"}, "p": {
            "IstimStart": "stim_start", 
            "IstimEnd": "stim_end", 
            "IstimAmplitude": "stim_amplitude", 
            "IstimPeriod": "stim_period", 
            "IstimPulseDuration": "stim_duration"
        }}, use_cython=True)

print m.__class__

s = pickle.dumps(m)
n = pickle.loads(s)

print m
print n