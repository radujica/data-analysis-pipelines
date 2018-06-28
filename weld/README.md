# Requirements
 * Weld, LLVM, pyweld, and grizzly; follow instructions here: https://github.com/weld-project/weld
 * Python 2.7
 * pipenv: https://github.com/pypa/pipenv

# Setup
    cd weld
    pipenv install -e <path-to-pyweld>
    pipenv install -e <path-to-grizzly>
    pipenv install

# Run
    pipenv run python pipeline.py
