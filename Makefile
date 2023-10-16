install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

lint:
	pylint --disable=R,C src/wrapper/run_analytical_sim.py
	pylint --disable=R,C src/wrapper/run_sim.py

format:
	black *.py
