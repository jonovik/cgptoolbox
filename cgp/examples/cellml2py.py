"""
Code generation service for cgptoolbox.

cellml = '''
<?xml version="1.0"?>
<model xmlns="http://www.cellml.org/cellml/1.0#" xmlns:cmeta="http://www.cellml.org/metadata/1.0#" xml:base="" cmeta:id="fitzhugh_arimoto_yoshizawa_1961" name="fitzhugh_arimoto_yoshizawa_1961">
   <units name="millisecond"><unit prefix="milli" units="second"/></units>
   <units name="per_millisecond"><unit exponent="-1" prefix="milli" units="second"/></units>
   <component name="Main">
      <variable cmeta:id="Main_t" name="t" units="millisecond"/>
      <variable cmeta:id="Main_w" initial_value="0" name="w" units="dimensionless"/>
      <variable name="I" units="dimensionless"/>
      <math xmlns="http://www.w3.org/1998/Math/MathML">
         <apply><eq/><ci>I</ci><cn xmlns:cellml="http://www.cellml.org/cellml/1.0#" cellml:units="dimensionless">0</cn></apply>
         <apply><eq/><apply><diff/><bvar><ci>t</ci></bvar><ci>w</ci></apply><cn xmlns:cellml="http://www.cellml.org/cellml/1.0#" cellml:units="per_millisecond">42</cn></apply>
      </math>
   </component>
</model>
'''
with closing(urllib.urlopen("http://bebiservice.umb.no/bottle/cellml2py", data=urllib.urlencode({"cellml": cellml}))) as f:
    print f.read()

url = "http://models.cellml.org/workspace/bondarenko_szigeti_bett_kim_rasmusson_2004/@@rawfile/99f4fd6804311c571a7143515003691ab2e430fb/bondarenko_szigeti_bett_kim_rasmusson_2004_apical.cellml"
with closing(urllib.urlopen(url)) as f:
    s = f.read()
with closing(urllib.urlopen("http://bebiservice.umb.no/bottle/cellml2py", data=urllib.urlencode(dict(cellml=s)))) as f:
    print f.read()
with closing(urllib.urlopen("http://bebiservice.umb.no/bottle/cellml2py/" + url)) as f:
    print f.read()

"""

import bottle

from cgp.physmod.cellmlmodel import generate_code

@bottle.get("/cellml2py/<url:path>")
def get(url):
    return generate_code(url)

@bottle.post("/cellml2py")
def post():
    return generate_code(bottle.request.forms.cellml.strip())

if __name__ == "__main__":
    bottle.run(debug=True, reloader=True)
