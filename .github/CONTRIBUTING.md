## Bug reports
Please include:
1. Which version you are using (`python -c "import tiler; print(tiler.__version__)"`)
2. A minimal code example to recreate the issue


## Code contribution
First off, thanks for taking the time!  
Please feel free to contact me with questions.

1. Fork the repo, then clone it and install in the editable mode:
```bash
git clone git@github.com:YOURNAME/tiler.git && cd tiler
pip install -e .[all]
```

2. Add your changes! 

3. After you've finished with your changes, please run linter, tests and check that coverage didn't go down too much:
```bash
black tiler tests
coverage run -m pytest
coverage report
```

4. If you have made any API and/or documentation changes, please regenerate docs by running the script:
```bash
cd misc
./docs.sh
```

5. Once you want to share what you've changed, please commit, push make a pull request to the main repo.

6. Github will run lint and tests, but please don't rely on that and test before pull request :)

