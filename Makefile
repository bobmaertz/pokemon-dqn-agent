.PHONY: venv lint requirements

v
	. venv/bin/activate 

update: venv 
	pip install --upgrade pip
	pip freeze > requirements.txt
    
lint: venv
	autopep8 -i --aggressive *.py

