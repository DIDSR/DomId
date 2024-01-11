# Developer hints

## Generate documentation with Sphinx

It is recommended to set up a separate Python virtual environment for the Sphinx run as it has additional dependencies beyond what's required by DomId (see `docs/requirements.txt`).

You should be able to generate documentation for DomID by running the following:

```
sh sh_gen_doc.sh
```

Sphinx will regenerate the documentation in html format, and you can find the output in the 'build' directory.

### Alternative approaches

- Alternatively, to rebuild the documentation, run the following command from the root of the repository:

```
sphinx-build -b html docs/ docs/build
```

- There is also Makefile in the project, the shortcut command to build the documentation is to run `make html` from the root of the repository (which is used by `sh_gen_doc.sh` above).

## Automated (unit) tests

For automated testing of this python package we use the [pytest](https://docs.pytest.org/en/latest/getting-started.html#getstarted) framework.
Tests are found in the `domid/tests/` directory.
Tests can be executed locally with:

```
poetry run pytest domid
```

## DomainLab

- To use the latest version of [DomainLab](https://github.com/marrlab/DomainLab) (rather than the version that DomId was tested with), run `git submodule update --remote`.
- By default DomId uses the master branch of DomainLab. If desired, you can set DomId to use another branch of DomainLab (replace `<branch_name>` with the name of the branch you want to use):

```
git config -f .gitmodules submodule.DomainLab.branch <branch_name>
git submodule update --remote
```

