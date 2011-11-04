.. _tool-characterization:

VPH ToolKit Tool Characterisation for the cGPtoolbox
====================================================

In reference to 
`VPH_ToolKit_Guideline_G01_Tool_Characterisation_1v0.pdf`, p. 16.

Tool information
----------------

Name
  cGPtoolbox
Version
  .. todo:: Insert version number here.
Function
  Workflow and building blocks for causally cohesive genotype-phenotype modelling.
Specialty
  Causally cohesive genotype-phenotype (cGP) modelling.
Input-output data
  Arrays with named fields. More complex data structures can be used, at the 
  cost of decreased generality.
Input-output data dimension
  From genotypes to phenotypes, where phenotypes include parameters from low to 
  high levels in the physiological hierarchy.
Licence
  Free and open, to be determined.
  
  .. todo:: Determine licence.

Certification
  None; this is currently just a research tool.

Tool specification
------------------

Language
  Python, interacting with other languages as necessary (for example: R for 
  statistical analysis, Sundials (in C) or OpenCMISS (in Fortran) for numerical 
  simulation, HDF for portable storage, Matlab for visualization, Excel for 
  summaries and interactive exploration).

  .. todo:: Add links.

OS
  Cross-platform.
Installation recommendation
  A scientific Python distribution. Separate installs for R, Sundials, and 
  other required software.
Third-party libraries
  N/A
Type of tool
  Software framework for causally cohesive genotype-phenotype modelling
Type of computation
  Flexible, including HPC for parallel computation of phenotypes from genotypes.

Tool description
----------------

Short purpose
  Workflow and building blocks for causally cohesive genotype-phenotype 
  modelling. We provide simple building blocks that are nevertheless usable for 
  non-trivial work. Each type of building block can have multiple 
  interchangeable options, which can be assembled into a cGP study. However, the 
  workflow is more general than the implementation; users may wish to roll their 
  own building blocks using other languages and/or data formats.
Documentation link
  Python docstrings and doctests are used throughout. Doctests provide code 
  examples and unit tests integrated with the documentation. Docstrings are 
  automatically processed by Doxygen, pydoc (Doxygen-like tool that ships with 
  Python) and the IPython interactive shell.
  
  .. todo:: Link to home.
  .. todo:: Demonstration modules.
  
Keywords
  causally cohesive genotype-phenotype modeling; 
  multivariate genotype-to-phenotype map; 
  cGP
Citation and reference papers
  .. todo::
     
     Cite papers:
     
     * Gjuvsland et al. JEB
     * Vik et al. subm. FGP
     * Wang et al. subm.
     * ...

Long purpose
  See :ref:`features`

Testing
  Python docstrings and doctests are used throughout. Doctests provide code 
  examples and unit tests integrated with the documentation. Docstrings are 
  automatically processed by Doxygen, pydoc (Doxygen-like tool that ships with 
  Python) and the IPython interactive shell.
  
  .. todo:: Link to demonstration modules.
  
Download links
  .. todo:: Link to downloads.

Tool context
------------

People involvement
  * Jon Olav Vik
  * Arne B. Gjuvsland
  * Yunpeng Wang
  * Nicolas P. Smith
  * Peter J. Hunter
  * Stig W. Omholt
Authors
  * Jon Olav Vik
  * Arne B. Gjuvsland
  * Yunpeng Wang
  * Nicolas P. Smith
  * Peter J. Hunter
  * Stig W. Omholt
Support
  Basic building blocks and demonstration modules will be supported, including 
  interfacing with supported repositories. Note that the work of end users will 
  typically include specific software and models that are beyond the scope of 
  cGPtoolbox support.
How many people involved
  One person full time, other authors contributing.
Reactivity
  No guarantees.
Type of collaboration
  VPH NoE Exemplar Project.
Funding status
  VPH NoE Exemplar Project 7; approximately one person-year (2011).
Institute/organization
  VPH NoE member institutions:
  
  * Cigene, Norwegian University of Life Sciences
  * Auckland Bioengineering Institute, New Zealand
  * King's College London, United Kingdom.

End-users target
  Researchers, paving the way for eventual clinical applications.
Development plan
  Streamline as a tool for education and exploratory cGP studies. 
  Further development priorities will be determined by user community response.
Website
  .. todo:: Publish on GitHub.
Use-case
  Demonstration modules.
  
  .. todo:: Demonstration modules.

Training and courses
  Demonstration modules.
Rights
  Free and open licence, details to be determined.
  
  .. todo:: Determine licence.

Tool functionality and speciality
---------------------------------

The cGP approach potentially encompasses all types of models described in the 
Model characterization guidelines (G02), Figure 1 (p. 12). We will illustrate 
the approach using a limited set of model types. The parameters of models and 
metamodels at multiple physiological levels are phenotypes of potential 
interest for a cGP modelling study. Ontogenies and semantic interoperability 
(G04) are important for model comparisons, automatic labelling and annotation 
of figures and analysis results.

.. todo:: Add links, including demo of semantic interoperability.

Supported data and model resources
  Markup languages:
  
  * CellML
  * SBML to follow depending on user response

  Model repositories
  
  * http://models.cellml.org
  * http://biomodels.org to follow depending on user response
  
  Genome databases
  
  * HapMap
  * more to follow depending on user response

Tool usability
--------------

Documentation and examples will be integrated into the source code in standard 
Python docstrings. Thus, function descriptions and call signatures are 
available from the interactive shell, facilitating prototyping. Code and 
demonstration modules will be kept Pythonic and concise to bring out the 
concepts and principles underlying the workflow.
