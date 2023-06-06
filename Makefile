install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

lint:
	pylint --disable=R,C src/wrapper/runAnalyticalSim.py
	pylint --disable=R,C src/wrapper/runSim.py

format:
	black *.py
