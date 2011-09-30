A toolbox for causally cohesive genotype-phenotype modeling
==========================================================================

.. figure:: cgpstudy.svg

A comprehensive understanding of how genetic variation causes phenotypic 
variation of a complex trait is a long-term disciplinary goal of genetics. The 
basic premise is that in a well-validated model that is capable of accounting 
for the phenotypic variation in a population, the causative genetic variation 
will manifest in the model parameters.

In this context, the term :term:`phenotype`
refers to any relevant measure of model behaviour, whereas the term 
:term:`parameter` denotes a quantity that is constant over the time-scale of the 
particular model being studied. However, even the lowest-level :doi:`model 
parameters are themselves phenotypes <10.1534/genetics.108.087064>`, 
whose genetic basis may be mono-, oligo- or polygenic, and whose physiological 
basis can be mechanistically modelled at ever deeper levels of detail.

The term :term:`causally cohesive genotype-phenotype modeling` (cGP modeling) 
thus denotes an 
approach where low-level parameters have an articulated relationship to the 
individual's genotype, and higher-level phenotypes emerge from the 
mathematical model describing the causal dynamic relationships between these 
lower-level processes. It aims to bridge the gap between standard population 
genetic models that simply assign phenotypic values directly to genotypes, and 
mechanistic physiological models without an explicit genetic basis. This 
forces a causally coherent depiction of the genotype-to-phenotype (GP) map.

.. contents::

Aims of the toolbox
==========================================================================

The cGPtoolbox aims to facilitate researchers' entry into cGP modeling by 
providing a cGP modelling framework in a population context, integrating and 
interfacing with existing VPH tools. The toolbox will provide the means for 
extensive explorative *in silico* studies as well as integration of 
patient-specific information in multiscale models to account for the 
individual’s genotype in the model parameterisation process. It adds to the 
VPH Toolkit by integrating genetic structure information, bioinformatic 
information and infrastructure and multiscale and multiphysics models and 
associated infrastructure. The strength of the cGP toolbox as a relevant 
research tool will be illustrated by specific examples of use:

* as an explorative tool for better understanding of key genetic concepts 
  like dominance, epistasis, pleiotropy, penetrance and expressivity
  in biologically realistic complex trait situations and in a 
  patient-specific perspective;
* to elucidate the fine structure of the distribution of individuals in a 
  high-dimensional phenotypic landscape associated with a pathological 
  condition as a function of genetic variation;
* as a test bed for developing new fine mapping methodologies within 
  statistical genetics aimed at exploiting high-dimensional phenotypic 
  information.

The cgptoolbox is a step towards providing computational tools for 
attaching GP maps of parameters to a multiscale modelling framework in 
order to handle patient-specific issues. We think this is an important 
delivery preparing for a future situation where acquisition of 
:doi:`high-dimensional phenotypic data <10.1038/nrg2897>` from patients 
become routine and the VPH community has come closer to its key goal 
of achieving more integration across multiple spatial and temporal scales.

.. figure:: workflow.png

   Simulation pipeline for causally cohesive genotype-phenotype studies. 
   Blue arrows denote functions that generate genotypes or transform them 
   through successive mappings, genotype to parameter to "raw" phenotypes to 
   aggregated phenotypes. The surrounding text exemplifies different 
   alternatives for each piece of the pipeline. "Virtual experiments" 
   interact with physiological models to generate phenotypes defined by the 
   system's response to external stimuli.


About the cgptoolbox 
==========================================================================

Genetics is defined as the :wiki:`science of genes, heredity, and the 
variation of organisms <genetics>`. Gaining a real understanding of the 
variation of organisms as a function of genes and environment in a 
mechanistic sense, i.e understanding the genotype-phenotype map (GP map) 
- is a tremendous challenge that awaits technological, conceptual and 
methodological breakthroughs. But this is where we have to go if we aim 
for a future genetics theory that bridges the genotype-phenotype gap by 
both generic and specific causal explanatory models. The impact of a 
mature genetics theory such as this on production biology, evolutionary 
biology and biomedicine can hardly be overstated. Recent breakthroughs 
concerning large-scale, high-throughput genotyping and phenotyping 
instrumentation and methodological means to model very complex biological 
structures based on lower-level processes suggest that it is not 
premature to make heuristic use of this vision in terms of research 
programme objectives. The establishment of such a theory will of 
necessity have to involve the extensive use of mathematics, statistics, 
informatics and biological physics guided by biological data in the 
broader sense, it will force new developments within these disciplines, 
and it will have to involve very advanced eInfrastructures. The VPH 
programme provides a very promising conceptual and methodological base 
for such a theory development given that it establishes a better 
interface with the theoretical as well as experimental genetics 
communities. We propose that one of the best ways to facilitate this 
development is to integrate with existing `VPH tools 
<http://toolkit.vph-noe.eu/>`_ a modelling framework which can handle GP 
map issues associated with multiscale models in a population context, and 
illustrate how it can be used by the genetics research community to 
address some of its key disciplinary issues otherwise beyond reach. A set 
of community tools facilitating the use of such a modelling framework is 
the planned core deliverable of this exemplar project.

Glossary
==========================================================================

.. glossary::
   :sorted:   

   :wiki:`phenotype`
      An organism's observable characteristics or traits.

   causally cohesive genotype-phenotype modeling
      A modeling approach where low-level parameters have an articulated 
      relationship to the individual's genotype, and higher-level phenotypes 
      emerge from the mathematical model describing the causal dynamic 
      relationships between these lower-level processes.




Contents
==========================================================================

.. toctree::
   :maxdepth: 2

.. automodule:: docutils.utils
    :members:


.. py:function:: enumerate(sequence[, start=0])
    
    Return an iterator that yields tuples of an index and and item of the *sequence*.

.. function:: publish(paper, impact_factor=123)
    
    Make the professor happy.

.. function:: test()

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

And remember, it is possible to :func:`publish` almost anything.
