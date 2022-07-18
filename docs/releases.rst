:orphan:

.. _releases:

..
    The following command allows to retrieve all commiters since a specified
    commit. From http://stackoverflow.com/questions/6482436/list-of-authors-in-git-since-a-given-commit
    git log 2e29eba.. --format="%aN <%aE>" --reverse | perl -e 'my %dedupe; while (<STDIN>) { print unless $dedupe{$_}++}'


========
Releases
========

Version 0.2
===========
| [FIX] Documentation and docker workflow file (#449)
| [RELEASE] Changes for release v0.2 (#446)
| [ADD] Allow users to pass feat types to tabular validator (#441)
| [ADD] docs for forecasting task (#443)
| [FIX] fit updates in gluonts (#445)
| [ADD] Time series forecasting (#434)
| [FIX] fix dist twine check for github (#439)
| [ADD] Subsampling Dataset (#398)
| [feat] Add __str__ to autoPyTorchEnum (#405)
| [ADD] feature preprocessors from autosklearn (#378)
| [refactor] Fix SparseMatrixType --> spmatrix and add ispandas (#397)
| [ADD] dataset compression (#387)
| [fix] Update the SMAC version (#388)
| [feat] Add new task inference for APT (#386)
| [FIX] Datamanager in memory (#382)
| [FIX] Fix: keyword arguments to submit (#384)
| [feat] Add coalescer (#376)
| [FIX] Remove redundant categorical imputation (#375)
| [ADD] scalers from autosklearn (#372)
| [ADD] variance thresholding (#373)
| [fix] Change int to np.int32 for the ndarray dtype specification (#371)
| [fix] Hotfix debug no training in simple intensifier (#370)
| [ADD] Test evaluator (#368)
| [FIX] Fix 361 (#367)
| [FIX] fix error after merge
| [ADD] Docker publish workflow (#357)
| [ADD] fit pipeline honoring API constraints with tests (#348)
| [FIX] Update workflow files (#363)
| [feat] Add the option to save a figure in plot setting params (#351)
| [FIX] Cleanup of simple_imputer (#346)
| [feat] Add an object that realizes the perf over time viz (#331)

Contributors v0.2
*****************

* Ravin Kohli
* Shuhei Watanabe
* Eddie Bergman
* Difan Deng

Version 0.1.1
==============
[refactor] Completely refactored version with a new scikit-learn compatible API.

Contributors v0.1.1
********************

* Ravin Kohli
* Shuhei Watanabe
* Francisco Rivera