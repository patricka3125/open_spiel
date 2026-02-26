# OpenSpiel

A framework for reinforcement learning and search/planning in games by Google DeepMind. Supports n-player zero-sum, cooperative, general-sum, sequential, simultaneous-move, perfect and imperfect information games. Core API and games are in C++ with Python bindings via pybind11.

## Build

- Prerequisites
  - Linux or macOS with `cmake` (>=3.17) and a C++17 compiler.
  - Python 3 (for Python bindings, enabled by default).
  - Abseil, nlohmann/json bundled as submodules under `open_spiel/abseil-cpp` and `open_spiel/json`.

- Default Build
  - `mkdir -p build && cd build`
  - `cmake ../open_spiel` (source dir is `open_spiel/`, **not** the repo root)
  - `cmake --build . -j$(nproc)`

- Build Types (via `BUILD_TYPE` env var)
  - `Debug` — `-g -Og`
  - `Testing` (default) — `-O2`, keeps all runtime checks
  - `Release` — `-O3 -DNDEBUG`, disables debug checks

- Shared Library Build
  - `BUILD_SHARED_LIB=ON cmake ../open_spiel` — builds `libopen_spiel.so`

- Using the Install Script
  - `./install.sh` (delegates to `open_spiel/scripts/install.sh`)
  - Full build + test: `open_spiel/scripts/build_and_run_tests.sh`

### Optional Dependencies

Controlled via environment variables (see `open_spiel/scripts/global_variables.sh`). Set before running CMake. Defaults are `OFF` unless noted.

| Variable | Default | Description |
|----------|---------|-------------|
| `OPEN_SPIEL_BUILD_WITH_PYTHON` | ON | Python bindings via pybind11 |
| `OPEN_SPIEL_BUILD_WITH_LIBTORCH` | ON* | PyTorch C++ API (libtorch) |
| `OPEN_SPIEL_BUILD_WITH_LIBNOP` | ON* | Serialization via libnop |
| `OPEN_SPIEL_BUILD_WITH_ACPC` | OFF | Universal Poker library |
| `OPEN_SPIEL_BUILD_WITH_HANABI` | OFF | Hanabi game |
| `OPEN_SPIEL_BUILD_WITH_JULIA` | OFF | Julia language bindings |
| `OPEN_SPIEL_BUILD_WITH_XINXIN` | OFF | xinxin Hearts program |
| `OPEN_SPIEL_BUILD_WITH_ROSHAMBO` | OFF | RoShamBo bots |
| `OPEN_SPIEL_BUILD_WITH_GAMUT` | OFF | GAMUT game generator |
| `OPEN_SPIEL_BUILD_WITH_ORTOOLS` | OFF | Google OR-Tools |
| `OPEN_SPIEL_ENABLE_JAX` | AUTO | JAX (Python, auto-detected) |
| `OPEN_SPIEL_ENABLE_PYTORCH` | AUTO | PyTorch (Python, auto-detected) |

\* Enabled in `global_variables.sh` but defaults to OFF in CMake if env var is not set.

## Testing

- CMake enables `ctest`. After building: `ctest --test-dir build --output-on-failure`
- Integration tests (Python): `open_spiel/integration_tests/`
- Playthrough regression tests: `open_spiel/integration_tests/playthrough_test.py`
  - Generate: `./open_spiel/scripts/generate_new_playthrough.sh <game_name>`
  - Regenerate all: `./scripts/regenerate_playthroughs.sh`

## Project Structure

```
open_spiel/                          # repo root
├── open_spiel/                      # main source (CMake source dir)
│   ├── CMakeLists.txt               # root CMake config
│   ├── spiel.h / spiel.cc           # core API: Game, State, GameType, GameInfo
│   ├── spiel_bots.h / .cc           # bot interfaces
│   ├── spiel_utils.h / .cc          # utility macros (SPIEL_CHECK_*)
│   ├── policy.h / .cc               # policy representations
│   ├── observer.h / .cc             # observation/info-state system
│   ├── game_parameters.h / .cc      # game parameter parsing
│   ├── algorithms/                  # C++ algorithm implementations
│   │   ├── alpha_zero_torch/        # AlphaZero with libtorch
│   │   ├── dqn_torch/              # DQN with libtorch
│   │   ├── cfr.h / .cc             # Counterfactual Regret Minimization
│   │   ├── mcts.h / .cc            # Monte Carlo Tree Search
│   │   └── ...                      # minimax, best_response, etc.
│   ├── games/                       # game implementations (C++)
│   ├── game_transforms/             # game wrappers/transforms
│   ├── bots/                        # bot implementations
│   ├── evaluation/                  # evaluation utilities
│   ├── utils/                       # shared utilities
│   ├── tests/                       # C++ test utilities
│   ├── examples/                    # C++ examples
│   ├── python/                      # Python code
│   │   ├── algorithms/              # Python algorithm implementations
│   │   ├── examples/                # Python examples
│   │   ├── games/                   # Python game implementations
│   │   └── tests/                   # Python tests
│   ├── scripts/                     # build/test/install scripts
│   │   ├── install.sh               # dependency installer
│   │   ├── build_and_run_tests.sh   # full build + test
│   │   └── global_variables.sh      # optional dependency flags
│   ├── abseil-cpp/                  # bundled Abseil
│   ├── json/                        # bundled nlohmann/json
│   ├── libnop/                      # optional: serialization
│   ├── libtorch/                    # optional: PyTorch C++ API
│   └── integration_tests/           # cross-game Python tests
├── docs/                            # documentation
│   ├── developer_guide.md           # adding games & algorithms
│   ├── install.md                   # installation guide
│   ├── concepts.md                  # API overview
│   └── contributing.md              # contribution guidelines
├── pybind11/                        # pybind11 submodule
└── examples/                        # top-level examples
```

## Core Architecture

### Key Abstractions (`spiel.h`)

- **`GameType`** — Static metadata: dynamics (sequential/simultaneous/mean-field), chance mode, information type, utility type, reward model.
- **`GameInfo`** — Instance metadata: action space size, player count, utility range, max game length.
- **`Game`** — Factory for states. Registered via `GameRegisterer`. Create with `open_spiel::LoadGame("game_name")`.
- **`State`** — Mutable game state. Key methods: `CurrentPlayer()`, `LegalActions()`, `ApplyAction()`, `IsTerminal()`, `Returns()`, `Clone()`.
- **`Observer`** — Observation/information-state tensor generation.

### Game Registration Pattern

Games self-register via a static `GameRegisterer` at the bottom of their `.cc` file:
```cpp
namespace {
const GameType kGameType{/* ... */};
std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::make_shared<MyGame>(params);
}
REGISTER_SPIEL_GAME(kGameType, Factory);
}
```

### Include Convention

Headers use the prefix `open_spiel/` — e.g. `#include "open_spiel/spiel.h"`. The parent of the `open_spiel/` directory is on the include path.

## Adding a New Game

1. Copy a reference game from `open_spiel/games/` (e.g. `tic_tac_toe.*`).
2. Rename classes, namespaces, header guards, and the short name.
3. Add source/test files to `open_spiel/games/CMakeLists.txt`.
4. Add short name to `open_spiel/python/tests/pyspiel_test.py`.
5. Implement `Game` and `State` methods per `spiel.h` docs.
6. Generate a playthrough: `./open_spiel/scripts/generate_new_playthrough.sh <name>`.
7. Run through cpplint / pylint for style conformance.

## Adding a New Algorithm

1. Add `.h` / `.cc` / `_test.cc` to `open_spiel/algorithms/`.
2. Register in `open_spiel/algorithms/CMakeLists.txt`.
3. For Python: mirror in `open_spiel/python/algorithms/`.

## Code Style

- **C++**: Google C++ Style Guide. Use `cpplint`. C++17 standard.
- **Python**: Google Python Style Guide. Use `pylint` with Google pylintrc.
- **Assertions**: Use `SPIEL_CHECK_*` macros (always active) and `SPIEL_DCHECK_*` (debug-only, disabled in Release).
- **Naming**: Games use `snake_case` short names. Classes use `PascalCase`.

## Language Bindings

- **Python** — Primary. Full pybind11 bindings in `open_spiel/python/`.
- **Julia** — `open_spiel/julia/` (active).
- **Rust** — `open_spiel/rust/` (unmaintained).
