#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  prep.sh — Conda environment setup for Photo Tagger
#
#  Usage:
#    ./prep.sh          # Create environment and install dependencies
#    ./prep.sh --exit   # Deactivate the conda environment
# ─────────────────────────────────────────────────────────────────────────────

CONDA_ENV_NAME="photo-tagger"
PYTHON_VERSION="3.12"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

# ── Check conda is available ──────────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
    echo -e "${RED}ERROR: conda not found.${RESET}"
    echo -e "Please install Miniconda or Anaconda first:"
    echo -e "  ${CYAN}https://docs.conda.io/en/latest/miniconda.html${RESET}"
    exit 1
fi

# Ensure conda shell functions are available (needed for both activate and deactivate)
eval "$(conda shell.bash hook)"

# ── Handle --exit flag ────────────────────────────────────────────────────────
if [[ "$1" == "--exit" ]]; then
    if [[ -n "$CONDA_DEFAULT_ENV" && "$CONDA_DEFAULT_ENV" != "base" ]]; then
        echo -e "${YELLOW}Deactivating conda environment '${CONDA_DEFAULT_ENV}'...${RESET}"
        conda deactivate
        echo -e "${GREEN}✓ Environment deactivated. Goodbye!${RESET}"
    else
        echo -e "${YELLOW}No active conda environment to exit.${RESET}"
    fi
    exit 0
fi

# ── Banner ────────────────────────────────────────────────────────────────────
echo -e "${BOLD}${CYAN}"
echo "  ┌─────────────────────────────────┐"
echo "  │       Photo Tagger Launcher     │"
echo "  └─────────────────────────────────┘"
echo -e "${RESET}"

CONDA_VERSION=$(conda --version 2>&1)
echo -e "  Conda          : ${CYAN}${CONDA_VERSION}${RESET}"

# ── Detect NVIDIA GPU ─────────────────────────────────────────────────────────
HAS_GPU=false
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    if [[ -n "$GPU_NAME" ]]; then
        HAS_GPU=true
        echo -e "  GPU            : ${GREEN}✓ ${GPU_NAME}${RESET}"
    fi
fi

if [[ "$HAS_GPU" == false ]]; then
    echo -e "  GPU            : ${YELLOW}None detected — will use CPU${RESET}"
fi

# ── Create conda environment if it doesn't exist ─────────────────────────────
if ! conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    echo -e "\n  ${YELLOW}Conda environment '${CONDA_ENV_NAME}' not found. Creating...${RESET}"
    conda create -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION" -y --quiet
    echo -e "  ${GREEN}✓ Environment created.${RESET}"
    NEEDS_INSTALL=true
else
    echo -e "  Environment    : ${GREEN}${CONDA_ENV_NAME} (exists)${RESET}"
    NEEDS_INSTALL=false
fi

# ── Activate the environment ──────────────────────────────────────────────────
conda activate "$CONDA_ENV_NAME"
echo -e "  Status         : ${GREEN}✓ Activated${RESET}"

# ── Install dependencies if needed ───────────────────────────────────────────
if [[ "$NEEDS_INSTALL" == true ]]; then
    echo -e "\n  ${YELLOW}Installing dependencies (this may take several minutes)...${RESET}\n"

    # Upgrade pip first
    pip install --upgrade pip --quiet

    # Remove stray conda packages that cause dependency noise
    conda remove -n "$CONDA_ENV_NAME" ipython ipykernel --quiet -y 2>/dev/null || true

    if [[ "$HAS_GPU" == true ]]; then
        echo -e "  ${CYAN}NVIDIA GPU detected — installing CUDA toolkit + cuDNN via conda...${RESET}"
        conda install -n "$CONDA_ENV_NAME" -c conda-forge cudatoolkit cudnn -y --quiet
        echo -e "  ${GREEN}✓ CUDA toolkit + cuDNN installed${RESET}"
        echo -e "  Installing ${CYAN}tensorflow (GPU)${RESET}..."
        pip install "tensorflow[and-cuda]==2.20.0" --quiet 2>&1 | grep -v "dependency conflicts" | grep -v "WARNING" || true
        echo -e "  ${GREEN}✓ tensorflow (GPU)${RESET}"
    else
        echo -e "  Installing ${CYAN}tensorflow (CPU)${RESET}..."
        pip install "tensorflow==2.20.0" --quiet 2>&1 | grep -v "dependency conflicts" | grep -v "WARNING" || true
        echo -e "  ${GREEN}✓ tensorflow (CPU)${RESET}"
    fi

    # Remaining packages
    for pkg in "deepface" "opencv-python" "tqdm" "tf-keras==2.20.1"; do
        echo -e "  Installing ${CYAN}${pkg}${RESET}..."
        pip install "$pkg" --quiet 2>&1 | grep -v "dependency conflicts" | grep -v "WARNING" || true
        echo -e "  ${GREEN}✓ ${pkg}${RESET}"
    done

    echo -e "\n  ${GREEN}✓ All dependencies installed.${RESET}"

else
    # Check for missing packages even if the env already exists
    MISSING=()
    declare -A PIP_NAMES=(
        ["deepface"]="deepface"
        ["opencv-python"]="opencv-python"
        ["tqdm"]="tqdm"
        ["tensorflow"]="tensorflow"
        ["tf-keras"]="tf-keras"
    )

    for pkg in "${!PIP_NAMES[@]}"; do
        pip_name="${PIP_NAMES[$pkg]}"
        if ! pip show "$pip_name" &>/dev/null; then
            MISSING+=("$pkg")
        fi
    done

    if [[ ${#MISSING[@]} -gt 0 ]]; then
        echo -e "\n  ${YELLOW}Missing packages detected. Installing: ${MISSING[*]}${RESET}\n"
        for pkg in "${MISSING[@]}"; do
            # Use pinned versions for tensorflow and tf-keras
            [[ "$pkg" == "tensorflow" ]] && pkg="tensorflow==2.20.0"
            [[ "$pkg" == "tf-keras"   ]] && pkg="tf-keras==2.20.1"
            echo -e "  Installing ${CYAN}${pkg}${RESET}..."
            pip install "$pkg" --quiet 2>&1 | grep -v "dependency conflicts" | grep -v "WARNING" || true
            echo -e "  ${GREEN}✓ ${pkg}${RESET}"
        done
    else
        echo -e "  Dependencies   : ${GREEN}✓ All present${RESET}"
    fi
fi

# ── Check for required scripts ────────────────────────────────────────────────
REQUIRED_SCRIPTS=("pt.py" "common.py" "enroll.py" "scan.py" "report.py")
MISSING_SCRIPTS=()
for script in "${REQUIRED_SCRIPTS[@]}"; do
    if [[ ! -f "$script" ]]; then
        MISSING_SCRIPTS+=("$script")
    fi
done

if [[ ${#MISSING_SCRIPTS[@]} -gt 0 ]]; then
    echo -e "\n  ${YELLOW}⚠  Missing scripts: ${MISSING_SCRIPTS[*]}${RESET}"
    echo -e "  Make sure all scripts are in the same folder as prep.sh.\n"
else
    echo -e "  Scripts        : ${GREEN}✓ All scripts found${RESET}"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo -e ""
echo -e "${BOLD}  Installation verified!${RESET}"
echo -e ""
echo -e "  ${CYAN}Example commands (once inside the environment):${RESET}"
echo -e "  ./pt.py enroll --known ./known_people --db faces.db"
echo -e "  ./pt.py scan   --photos ./my_photos   --db faces.db --output results.json"
echo -e "  ./pt.py report --output results.json"
echo -e ""
echo -e "  ${YELLOW}To exit the environment later, run:  ${BOLD}conda deactivate${RESET}"
echo -e ""

# ── Prompt user to enter the environment ─────────────────────────────────────
echo -e "${BOLD}  Setup complete! To enter the environment run:${RESET}"
echo -e ""
echo -e "  ${GREEN}conda activate ${CONDA_ENV_NAME}${RESET}"
echo -e ""
