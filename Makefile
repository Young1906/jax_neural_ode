dev:
	python -m lib.neural_ode --config configs/node.yml


fmt_code:
	find lib -type f -name "*.py" | xargs reorder-python-imports;
	find lib -type f -name "*.py" | xargs black;

