cgptoolbox features
==========================================================================

.. todo:: Some of this is well covered by other projects, e.g. SimuPop.

**Virtual genome data structure**, locating genes and markers on 
chromosomes, keeping track of physical and genetic map units, and 
providing slots for user-defined parameter data, which can be accessed 
by the physiological models.

**Importing genomic data from public databases** (e.g. the HapMap 
project and Entrez databases such as SNP and Gene) into virtual 
genomes. This will be done with the existing module Biopython.

**Meiosis of virtual genomes**, taking into account chromosomal 
arrangement and recombination rates for both markers and functional 
genes.

**Functions for dealing with population structure** and observed or 
model-generated pedigrees.

**Core functionality for doing population-level simulations** combining 
structural genome dynamics (keeping track of recombination, allele 
frequencies and haplotype block structures) with cGP models (in 
addition to the traditional GP models from quantitative genetics). The 
software will be designed to be modular such that cGP models and 
pedigree structures can be easily changed. Examples will span the 
range from cellular models in CellML (see below) to whole-organ 
simulations of continuum dynamics using openCMISS.

**Routines for turning CellML models into cGP models**. This will be 
done with as little manual work as possible, with automatic download 
from the CellML repository and integration using the CVODE solver.

**Design patterns for :term:`virtual experiments`** that interact with 
physiological models to generate phenotypes defined by the system's 
response to external stimuli. For instance, a given pacing protocol 
can be applied to a whole class of heart cell models.

**Setting up simulations based on publicly available genomic data**
from the HapMap project and Entrez databases such as SNP and Gene, 
using Biopython.

**Export routines** to data formats for state-of-the-art quantitative 
genetic software for doing heritability estimates, haplotype block 
detection and genome-wide association studies.

**Convenient packaging into tasks** that can be run trivially in 
parallel on computer clusters, automatically consolidating results as 
they become available.
