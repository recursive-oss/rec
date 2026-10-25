.PHONY: test test-cli test-engine

# The OpenCode client is built for Apple Silicon only.
ifeq ($(shell uname -sm),Darwin arm64)
test: test-cli test-engine
CLIENTS = opencode pi
else
test: test-cli
CLIENTS = pi
endif

test-cli:
	go test -race ./...
	go vet ./...
	node --test tests/pi_extension_test.mjs tests/pi_logo_test.mjs
	bash -n install.sh
	python3 -m py_compile packaging/*.py tests/*.py tests/data/*.py
	go build -o .build/rec ./cli
	python3 tests/launch_client_test.py .build/rec $(CLIENTS)
	python3 tests/installer_test.py
	python3 tests/package_test.py

PYTHON_ROOT = $(shell uv python dir)/cpython-3.12.12-macos-aarch64-none

test-engine:
	cd engine && uv venv --allow-existing --python 3.12 && uv pip install '.[validation]'
	cd engine && nice -n 10 .venv/bin/pytest -v -rs -n auto --dist loadgroup
	uv python install 3.12.12
	rm -rf .build/bundle-mlx .build/smoke
	$(PYTHON_ROOT)/bin/python3.12 packaging/build-mlx-runtime.py --python-root $(PYTHON_ROOT) --output .build/bundle-mlx
	python3 tests/build_smoke_runtime.py --runtime .build/bundle-mlx --output .build/smoke
	python3 packaging/package.py --runtime-dir .build/smoke/runtime --manifest .build/smoke/smoke-mlx.json --output .build/rec-smoke
	python3 tests/smoke_test.py .build/rec-smoke
	go build -o .build/rec ./cli
	python3 tests/protocol_test.py .build/bundle-mlx/python/bin/python3.12 .build/rec
