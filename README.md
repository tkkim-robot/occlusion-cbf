# Occlusion CBF

This repository implements Occlusion CBF for safe control in obstacle environments.

## Installation

This project uses `uv` for dependency management.

1.  **Clone the repository**:
    ```bash
    git clone --recursive https://github.com/tkkim-robot/occlusion-cbf.git
    cd occlusion-cbf
    ```
    If you already cloned it without `--recursive`, run:
    ```bash
    git submodule update --init --recursive
    ```

2.  **Install dependencies**:
    ```bash
    uv sync
    ```

3.  **Run Tests**:
    ```bash
    uv run examples/test_occlusion.py
    ```
