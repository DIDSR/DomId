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

## Code style and formatting

This project uses [Black](https://black.readthedocs.io/en/stable/) for code formatting, with settings as specified in the `pyproject.toml` file. To run automated code formatting, from the base directory of the project execute:

```
bash sh_format_code.sh
```

## Using DomId with a development version of DomainLab

By default the official [DomainLab](https://github.com/marrlab/DomainLab) release from PyPI will be used. However, you can choose to use a more recent development versions of DomainLab from the Github repository. First, you will need to edit the `pyproject.toml` file accordingly, for example, along the lines of:

```
domainlab = {git = "https://github.com/marrlab/DomainLab.git", rev = "master"}
```

or

```
domainlab = { path = "./DomainLab", develop = true}
```

Additional hints:
- To use the latest version of [DomainLab](https://github.com/marrlab/DomainLab), it may help to run `git submodule update --remote`.
- If desired, you can set DomId to use another branch of DomainLab (replace `<branch_name>` with the name of the branch you want to use):

```
git config -f .gitmodules submodule.DomainLab.branch <branch_name>
git submodule update --remote
```

