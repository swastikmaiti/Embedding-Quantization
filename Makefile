install:
		pip install --upgrade pip &&\
			pip install -r requirements.txt
test:

format:
		black *.py
		black *.ipynb
lint:
		pylint --disable=R,C

all:
		install test