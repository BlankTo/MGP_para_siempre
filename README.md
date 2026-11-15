# MGP_para_siempre

Experiments (Studies 1–8 and the workspace grid) and the analyses that produce the paper's tables and figures.

## Install

Python 3.14 (tested).

```
pip install -r requirements.txt
```

## Run

```
python study1.py                 # run or resume one study
python study1.py --test 20       # time 20 sample tasks, nothing saved
python study1.py --info          # tasks done / pending
python run_all.py                # all studies, then all visuals
python run_all.py --only study1,study2 --no-visuals
python list_runs.py [study] [--detail]
```

Common options: `--workers N`, `--memory-mb M`.

Each study writes to `runs/<study>__<token>__<hash>.sqlite`:
- rerunning resumes and skips completed tasks;
- more repetitions or pools add only the new tasks to the same file;
- changing an identity parameter (grid levels, engine settings, Revisit IC or single archive limits) creates a new file.

## Visuals

```
python visuals/all.py
python visuals/res_group_n.py --help
```

Output goes to `output/` and the cache to `work/`. By default each visual reads the most recently updated execution of its study.
- `--input NAME` reads a specific execution instead of the most recent one. NAME is a file in `runs/`, with or without `.sqlite`, or a path, e.g. `--input study1__ric64x512b64__4a2e7b7f`. Visuals that read several studies take one flag per study instead: `--study2 NAME`, `--study6 NAME`, and so on.
- `--out DIR` writes to another folder.

## Parameters

- `config.py` holds the base values shared by every script: targets, strategies, grid levels, Revisit IC limits, archive limits, workers, memory and progress logging.
- Each `studyN.py`, `workspace.py` and `visuals/*.py` starts with its own parameter block. `DEFAULT` takes the value from `config.py`. Any other value overrides it for that script only.
