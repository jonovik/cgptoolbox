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

   Document successful installation as root user using 
   `Ubuntu <http://www.ubuntu.com/download/ubuntu/download>`_'s
   package system. Ubuntu also works well under 
   `VirtualBox <http://www.virtualbox.org/>`_
   on 64-bit Windows 7 and probably other platforms.

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
