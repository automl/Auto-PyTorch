:orphan:

.. _releases:

..
    The following command allows to retrieve all commiters since a specified
    commit. From http://stackoverflow.com/questions/6482436/list-of-authors-in-git-since-a-given-commit
    git log 2e29eba.. --format="%aN <%aE>" --reverse | perl -e 'my %dedupe; while (<STDIN>) { print unless $dedupe{$_}++}'


========
Releases
========

Version 0.1.0
==============
[refactor] Initial version of the new scikit-learn compatible API.
