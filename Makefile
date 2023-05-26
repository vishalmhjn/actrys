install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

lint:
	pylint --disable=R,C src/wrapper/runAnalyticalSim.py

format:
	black *.py
