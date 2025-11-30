.PHONY: run logs stop help

# Run the main experiment
run:
	@./run_script.sh experiments.main

# Watch the logs
logs:
	@tail -f logs/experiments_main.log

# Stop the running experiment
stop:
	@./stop_script.sh experiments.main

# Show help
help:
	@echo "Available commands:"
	@echo "  make run   - Run experiments.main"
	@echo "  make logs  - Watch the experiment logs"
	@echo "  make stop  - Stop the running experiment"
	@echo "  make help  - Show this help message"
