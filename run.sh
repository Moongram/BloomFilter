#!/bin/bash

PYTHON=$(command -v python3)

if [ -z "$PYTHON" ]; then
    PYTHON=$(command -v python)
    if [ -z "$PYTHON" ]; then
        echo "Error: Python 3 is not installed or not in the PATH"
        exit 1
    fi
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

"$PYTHON" "$SCRIPT_DIR/BF.py"