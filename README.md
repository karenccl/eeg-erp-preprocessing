# eeg-erp-preprocessing

Automated, reproducible preprocessing of event-related EEG in MATLAB (EEGLAB + ERPLAB). One script runs the full pipeline for every subject and writes cleaned epochs, averaged ERPs, QC logs, IC-review figures and trial-level tables.

## Features

- **Bad-channel detection after re-referencing** (flatline, robust deviation, RANSAC correlation; PREP-style), so electrodes next to the online reference are not flagged wrongly.
- **OPTICAT-style ICA**: training on a 2 Hz high-passed branch with overweighted saccade-onset segments; weights applied to a 0.1 Hz ERP branch.
- **Ocular and artefact IC detection** combining ICLabel, EOG correlation and saccade/fixation variance ratio, with a manual override file.
- **Two references from one ICA**: average reference (primary) and linked mastoids (derived from the same cleaned epochs and rejection flags).
- **Trial-level output** with optional single-trial ROI amplitudes and standardized measurement error (SME) per bin.
- **Graceful degradation**: no EOG channels or no mastoids are handled by skipping the dependent steps and logging a QC flag.
- **Reproducible**: fixed random seed per subject; serial or parallel execution.

## Pipeline

```
load → bad-channel detection → two branches:
    ERP branch (0.1 Hz HP)        ICA branch (2–100 Hz)
    resample, interpolate, average reference (both)
→ bins (ERPLAB) → OPTICAT ICA → IC classification → remove ICs from ERP branch
→ 30 Hz low-pass → epoch → baseline → artefact flagging
→ linked-mastoid copy → trial table + SME → average and save
```

## Requirements

- MATLAB R2019b or newer (developed with R2023b), Signal Processing Toolbox; Parallel Computing Toolbox only if `cfg.nWorkers > 0`
- [EEGLAB](https://sccn.ucsd.edu/eeglab/) 2023.1 with plugins: ERPLAB 10.11, ICLabel 1.7, clean_rawdata, and the import plugin for your file format (Curry, BrainVision or BIOSIG)

## Quick start

1. Open `eeg_preprocessing_pipeline.m` and edit the **USER SETTINGS** block:
   - `cfg.dataDir`, `cfg.outputRoot`, `cfg.filePattern` (`*.cdt`, `*.vhdr`, `*.bdf`, `*.edf` or `*.set`)
   - `cfg.stimCodes`, and either `cfg.binDescriptorPath` (Binlister bin descriptor file) or `cfg.eventlistPath` (ERPLAB event list)
   - `cfg.eogChans`, `cfg.mastoidChans`, `cfg.dropChans` to match your montage
   - `cfg.chanlocFile` if your recordings have no electrode coordinates
2. Set `cfg.subjectsToRun` to one or two subjects and run the script. Check `qc_logs/` and `ic_review/`.
3. Set `cfg.subjectsToRun = {}` to process everyone.

## Outputs

| Folder | Contents |
|---|---|
| `epochs/` | `<subj>_preprocessed.set`: cleaned epochs, average reference |
| `erp_avgref/` | `<subj>_avg.erp`: averaged ERP, average reference |
| `erp_linkedmastoid/` | `<subj>_avg_lm.erp`: averaged ERP, linked mastoids |
| `qc_logs/` | `<subj>_qc.json`: bad channels, ICs removed, trials per bin, QC flags |
| `ic_review/` | ICA decomposition, IC metrics, IC topographies, optional `<subj>_ic_decisions.csv` |
| `trial_level/` | `<subj>_trials.csv` and `<subj>_SME.csv` |

### Manual IC override

After reviewing `ic_review/<subj>_IC_topos.png`, create `ic_review/<subj>_ic_decisions.csv` with columns `ic` and `remove` (1 = remove, 0 = keep) and re-run that subject. The saved ICA is reused if the channel set and bad channels are unchanged.

## Notes

- Defaults (filters, thresholds, epoch window of -200 to 1000 ms, example ROI measures) are starting points; adjust them for your paradigm.
- Data are assumed to be in microvolts with 3-D electrode coordinates available.

## License

MIT. Author: Karen C.L. Chan.
