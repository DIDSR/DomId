# Developer hints

## Generate documentation with Sphinx

Probably set up a separate Python virtual environment. Then run the following:

```
sh gen_doc.sh
```

## DomainLab

- To use the latest version of [DomainLab](https://github.com/marrlab/DomainLab) (rather than the version that DomId was tested with), run `git submodule update --remote`.
- By default DomId uses the master branch of DomainLab. If desired, you can set DomId to use another branch of DomainLab (replace `<branch_name>` with the name of the branch you want to use):

```
git config -f .gitmodules submodule.DomainLab.branch <branch_name>
git submodule update --remote
```
