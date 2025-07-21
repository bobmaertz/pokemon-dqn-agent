.PHONY: venv lint requirements

venv:
	. venv/bin/activate 

update: venv 
	pip install --upgrade pip
	pip freeze > requirements.txt
    
lint: venv
	autopep8 -i --aggressive *.py

build: 
	docker build . -t pokemon_blue:latest 

run: 
	docker run \
		-v env_state:/workspace/env_state pokemon_blue:latest  \
			--replay_memory_size 300   \
			--steps_per_episode 10000   \
			--num_episodes 100   \
			--rom_path ./POKEMONR.GBC  \
			--state_file ./env_state/pokedex.state   \
			--model_name pokemon_blue_dqn   \
			--wandb_entity r-maertz-bobs-tools- \   
			--wandb_project PokemonRed  \
			--emulation_speed 32 \

