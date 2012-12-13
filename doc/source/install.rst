.. Testing cross-references:

.. _installation:

Installing the cgptoolbox
=========================

.. highlight:: bash

.. This is a comment. In fact, any directive that ReST cannot make sense of
   is considered a comment. Furthermore, ReST is picky about indentation. 
   In the enumerated lists below, the indentation of continuation lines must 
   exactly match the beginning of the line, not counting the number and dot. 
   There must be a blank line before further paragraphs in an item, and before 
   nested lists.

.. todo::

   Describe installation process in a typical HPC environment, non-python dependencies (R, hdf5,
   zeromq,) and file of required/recommended python packages for pip installation.
   
Installing on `Ubuntu <http://www.ubuntu.com/download/ubuntu/download>`_ 12.10 with root access
-----------------------------------------------------------------------------------------------
Installing cgptoolbox on the latest version of Ubuntu is relatively easy and Ubuntu also
works well under `VirtualBox <http://www.virtualbox.org/>`_ on 64-bit Windows 7 and other platforms. 
Most dependencies such as `hdf5 <http://www.hdfgroup.org/HDF5/>`_, `R <http://r-project.org>`_
and various `Python <http://python.org>`_ packages are in the Ubuntu repositories an can 
be installed with apt-get. The code line will install all dependencies.

.. code-block:: bash

   sudo apt-get install python-tables python-networkx python-pip python-rpy2 python-joblib cython

For solving differential equations the cgptoolbox utilizes the `SUNDIALS <http://www.llnl.gov/CASC/sundials>`_ 
library and the python bindings provided by the Python package `pysundials <http://pysundials.sourceforge.net>`_.
Ubuntu 12.10 offer SUNDIALS v.2.5.0, but the stable release of `pysundials <http://pysundials.sourceforge.net>`_ is for
v.2.3.0. The following lines of code adds the repository and signing key for Debian stable, installs libraries and header 
files for the 2.3.0 version of SUNDIALS and protect the packages from upgrading.

.. code-block:: bash

   sudo add-apt-repository 'deb http://ftp.de.debian.org/debian squeeze main'
   sudo apt-get install debian-archive-keyring					#not tested properly yet
   sudo apt-get install libsundials-serial=2.3.0-2 libsundials-serial-dev=2.3.0-2
   echo "libsundials-serial hold" | sudo dpkg --set-selections
   echo "libsundials-serial-dev hold" | sudo dpkg --set-selections

Once the right version of SUNDIALS is installed, pysundials and cgptoolbox can be installed 
with `pip <http://www.pip-installer.org/>`_ directly from their code repositories.

.. code-block:: bash

   sudo pip install svn+https://pysundials.svn.sourceforge.net/svnroot/pysundials/branches/2.3.0/@74
   sudo pip install git+https://github.com/jonovik/cgptoolbox.git
   
Installing on Linux witout root access 
--------------------------------------
On a typipcal HPC cluster the cgptoobox must be installed without root access. Before the installing the required Python 
packages you should check the following software dependencies and install (or ask the system administratir to install)  
missing pieces, version numbers refer to version that we have tested, other version could also work:

* `Python <http://python.org>`_  versions 2.7.3, 2.7.2. 
* `virtualenv <http://www.virtualenv.org>`_ version 1.8.2
* `R <http://www.r-project.org/>`_ , version 2.15.1. R must be built as a library (instructions below).
* `hdf5 <http://www.hdfgroup.org/HDF5/>`_ , version 1.8.7, 1.8.9.
* `SUNDIALS <http://www.llnl.gov/CASC/sundials>`_ 2.3.0 built as shared library (instructions below)

SUNDIALS 
^^^^^^^^
Download version 2.3.0 of the `SUNDIALS <http://www.llnl.gov/CASC/sundials>`_ library not the newest 2.5.0, 
since `pysundials <http://pysundials.sourceforge.net>`_ does not work for the latest version yet. 

.. code-block:: bash

   export INSTALLDIR=$HOME/usr	 
   tar -xzf sundials-2.3.0.tar.gz
   cd sundials-2.3.0
   ./configure --prefix=$INSTALLDIR --enable-shared --with-ldflags=-no-undefined
   make
   make install
   cd ~
   export CPATH=$INSTALLDIR/include:$CPATH

R - build as library
^^^^^^^^^^^^^^^^^^^^
.. code-block:: bash

   #download tarball from mirror and extract
   wget http://cran.uib.no/src/base/R-2/R-2.15.2.tar.gz
   tar xzf R-2.15.2.tar.gz						

   #configure, compile and install
   cd R-2.15.2
   export INSTALLDIR=$HOME/usr
   ./configure --prefix=$INSTALLDIR --enable-R-shlib
   make
   make install
   cd ~

   #set up environment variables
   export LD_LIBRARY_PATH=$INSTALLDIR/lib64/R/lib:$LD_LIBRARY_PATH
   export CPATH=$INSTALLDIR/lib64/R/include:$CPATH
   export PATH=$INSTALLDIR/bin:$PATH


   
Virtualenv with required python packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   #create and activate virtual Python environment
   virtualenv cgp			
   source cgp/bin/activate
   
   #install dependencies and cgptoolbox
   pip install "ipython>=0.12.1"
   pip install "numpy>=1.6.1"
   pip install "scipy>=0.10.1"
   pip install "matplotlib>=1.1.1"
   pip install "numexpr>=2.0.1"
   pip install "Cython>=0.16"
   pip install "tables>=2.3.1"		#requires hdf5 library
   pip install "joblib>=0.6.4"
   pip install "bottle>=0.11.4"
   pip install "pyzmq>=2.2.0"		
   pip install "rpy2>=2.2.6"		#requires R built as library
   pip install "networkx>=1.7"
   pip install svn+https://pysundials.svn.sourceforge.net/svnroot/pysundials/branches/2.3.0/@74
   pip install git+https://github.com/jonovik/cgptoolbox.git
   
   #packages for unittests and building documentation
   pip install "Sphinx>=1.1.3"
   pip install "nose>=1.2.1"
   


Building the documentation
--------------------------

Make sure you have `Sphinx >= 1.1 <http://sphinx.pocoo.org/latest/>`_ installed, 
then run the following from the ``cgptoolbox/doc`` directory::

   sphinx-apidoc -T -H cgptoolbox -o source ..
   make html

Documentation will end up in ``cgptoolbox/doc/build/html``.

Here's a one-liner to make both html and latex, ignoring any latex errors::

   time ((make clean; make html; make latex; cd build/latex; echo R | pdflatex cgptoolbox.tex; echo R | pdflatex cgptoolbox.tex; xdg-open cgptoolbox.pdf; xdg-open ../html/index.html) > all.txt 2>&1)

.. rubric:: If a plain ``make html`` fails:

For a pristine build, you can try wiping all doc files that are 
not under version control.

.. warning::
   
   You will probably want a dry run first to see 
   what will be removed. Run this from your ``cgptoolbox/doc`` directory::
   
      git clean -n -d -x
   
   Then, if you are certain that no important work will be lost::
   
      git clean -f -d -x
      sphinx-apidoc -T -H cgptoolbox -o source ..
      make html

Please report errors or omissions to jonovik@gmail.com.
