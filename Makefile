# ################################################################################
# # Configuration and Variables
# ################################################################################
ZIG    ?= $(shell which zig || echo ~/.local/share/zig/0.15.1/zig)
BUILD_TYPE    ?= Debug
BUILD_OPTS      = -Doptimize=$(BUILD_TYPE)
JOBS          ?= $(shell nproc || echo 2)
SRC_DIR       := src
BUILD_DIR     := zig-out
CACHE_DIR     := .zig-cache
BINARY_NAME   := zigformer-cli
RELEASE_MODE := ReleaseSafe
TEST_FLAGS := --summary all #--verbose
JUNK_FILES := *.o *.obj *.dSYM *.dll *.so *.dylib *.a *.lib *.pdb temp/

SHELL         := /usr/bin/env bash
.SHELLFLAGS   := -eu -o pipefail -c

################################################################################
# Targets
################################################################################

.PHONY: all help build rebuild run run-elz test test-elz release clean lint format docs serve-docs install-deps setup-hooks test-hooks
.DEFAULT_GOAL := help

help: ## Show the help messages for all targets
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*## .*$$' Makefile | \
	awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

all: build test lint docs  ## build, test, lint, and doc

init: ## Initialize a new Zig project
	@echo "Initializing a new Zig project..."
	@$(ZIG) init

build: ## Build project (e.g. 'make build BUILD_TYPE=ReleaseSafe')
	@echo "Building project in $(BUILD_TYPE) mode with $(JOBS) concurrent jobs..."
	@$(ZIG) build $(BUILD_OPTS) -j$(JOBS)

rebuild: clean build  ## clean and build

run: ## Run a Zig example (e.g. 'make run EXAMPLE=e1_ffi_1')
	@if [ "$(EXAMPLE)" = "all" ]; then \
	   echo "--> Running all Zig examples..."; \
	   fail=0; \
	   for ex in $(ZIG_EXAMPLES); do \
	      echo ""; \
	      echo "--> Running '$$ex'"; \
	      $(ZIG) build run-$$ex $(BUILD_OPTS) || { echo "FAILED: $$ex"; fail=1; }; \
	   done; \
	   exit $$fail; \
	else \
	   echo "--> Running Zig example: $(EXAMPLE)"; \
	   $(ZIG) build run-$(EXAMPLE) $(BUILD_OPTS); \
	fi

run-elz: build ## Run a Lisp example (e.g. 'make run-elz ELZ_EXAMPLE=e1-cons-car-cdr')
	@if [ "$(ELZ_EXAMPLE)" = "all" ]; then \
	   echo "--> Running all Lisp examples..."; \
	   fail=0; \
	   for ex in $(ELZ_EXAMPLES); do \
	      echo ""; \
	      echo "--> Running '$$ex'"; \
	      ./zig-out/bin/elz-repl --file $$ex || { echo "FAILED: $$ex"; fail=1; }; \
	   done; \
	   exit $$fail; \
	else \
	   echo "--> Running Lisp example: $(ELZ_EXAMPLE)"; \
	   ./zig-out/bin/elz-repl --file examples/elz/$(ELZ_EXAMPLE).elz; \
	fi

repl: ## Start the REPL
	@echo "Starting the REPL..."
	@$(ZIG) build repl $(BUILD_OPTS)

test: ## Run tests
	@echo "Running tests..."
	@$(ZIG) build test $(BUILD_OPTS) -j$(JOBS) $(TEST_FLAGS)

test-elz: ## Run Element 0 standard library tests
	@echo "Running Element 0 standard library tests..."
	@$(ZIG) build test-elz $(BUILD_OPTS) -j$(JOBS) $(TEST_FLAGS)

release: ## Build in Release mode
	@echo "Building the project in Release mode..."
	@$(MAKE) build BUILD_TYPE=$(RELEASE_MODE)

clean: ## Remove docs, build artifacts, and cache directories
	@echo "Removing build artifacts, cache, generated docs, and junk files..."
	@rm -rf $(BUILD_DIR) $(CACHE_DIR) $(JUNK_FILES) docs/api public

lint: ## Check code style and formatting of Zig files
	@echo "Running code style checks..."
	@$(ZIG) fmt --check $(SRC_DIR)

format: ## Format Zig files
	@echo "Formatting Zig files..."
	@$(ZIG) fmt .

docs: ## Generate API documentation
	@echo "Generating API documentation..."
	@$(ZIG) build docs

serve-docs: ## Serve the generated documentation on a local server
	@echo "Serving API documentation locally..."
	@cd docs/api && python3 -m http.server 8000

install-deps: ## Install system dependencies (for Debian-based systems)
	@echo "Installing system dependencies..."
	@sudo apt-get update
	@sudo apt-get install -y make llvm snapd
	@sudo snap install zig --beta --classic

setup-hooks: ## Install Git hooks (pre-commit and pre-push)
	@echo "Setting up Git hooks..."
	@if ! command -v pre-commit &> /dev/null; then \
	   echo "pre-commit not found. Please install it using 'pip install pre-commit'"; \
	   exit 1; \
	fi
	@pre-commit install --hook-type pre-commit
	@pre-commit install --hook-type pre-push
	@pre-commit install-hooks

test-hooks: ## Test Git hooks on all files
	@echo "Testing Git hooks..."
	@pre-commit run --all-files --show-diff-on-failure
