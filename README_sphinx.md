# Here is a quick guide how to generate documentation for DomID:

In order to rebuild the documentation, run the following command from the root of the repository: 

```sphinx-build -b html docs/ docs/build```

After running the command, Sphinx will regenerate the documentation in the specified format, and you can find the output in the 'build' directory.

There is Makefile in the project, the shortcut command to build the documentation is to run ```make html``` from the root of the repository.