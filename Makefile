.PHONY: run logs stop token token-logs token-stop help

# Run the main experiment
run:
	@./run_script.sh experiments.main

# Run token length analysis experiment
token:
	@./run_script.sh experiments.token_length_analysis_v2

# Watch the main experiment logs
logs:
	@tail -f logs/experiments_main.log

# Watch token length analysis logs
token-logs:
	@tail -f logs/experiments_token_length_analysis_v2.log

# Stop the running main experiment
stop:
	@./stop_script.sh experiments.main

# Stop the running token length analysis experiment
token-stop:
	@./stop_script.sh experiments.token_length_analysis_v2

# Show help
help:
	@echo "Available commands:"
	@echo "  make run         - Run experiments.main"
	@echo "  make token       - Run token length analysis experiment"
	@echo "  make logs        - Watch the main experiment logs"
	@echo "  make token-logs  - Watch token length analysis logs"
	@echo "  make stop        - Stop the running main experiment"
	@echo "  make token-stop  - Stop the token length analysis experiment"
	@echo "  make help        - Show this help message"
