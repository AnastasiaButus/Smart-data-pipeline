# Claude Code — Project Permissions

## Project root
`D:\Projects\Data_Agent_Project\smart-data-pipeline\`

---

## Auto-approved (no confirmation needed)

- Read and write any file inside the project root
- Create and delete files/directories inside the project root
- Run `pytest`, `python`, `pip` via `.venv` (e.g. `.venv/Scripts/python`, `.venv/Scripts/pytest`)
- `git add`, `git commit`, `git push`, `git status`, `git diff`, `git log`
- Install packages into the project `.venv` (pip install inside venv)

## Require explicit user confirmation

- Any file operation outside `D:\Projects\`
- Modifications to Windows system files or registry
- Global `pip install` without `--user` or outside a virtualenv
- Deleting or modifying `.git/`
- Force-push (`git push --force`) to any branch
- Running shell commands that affect system state outside the project (e.g. `rm -rf`, `taskkill`, `sc`, `reg`)

---

## Code conventions (always apply)

- All Python code: docstrings, type hints, `loguru` for logging
- Config values from `config.yaml` or `.env` only — no hardcoded values
- Use `pathlib.Path` for all file paths
- Follow existing module structure (see `src/`)
