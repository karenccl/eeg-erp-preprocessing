%% EEG PREPROCESSING PIPELINE FOR ERP DATA (EEGLAB + ERPLAB)
% Automated, reproducible preprocessing of event-related EEG: one script, one ICA, two reference
% schemes. Every subject gets a QC log, IC-review figures and single-trial/SME tables.
%
% Author:  Karen C.L. Chan <chanchinglam2430@gmail.com>
% License: MIT
%
% -------------------------------------------------------------------------------------------
% WHAT IT DOES
%   0  Load recording (.cdt, .vhdr, .bdf/.edf, .set or custom loader).
%   1  Keep bipolar EOG (not re-referenced, not in ICA, not used for trial rejection); drop other
%      non-EEG channels.
%   2  Saccade/blink detection on raw EOG (velocity-based, median-SD threshold). Skipped if no EOG.
%   3  Bad-channel detection on REFERENCED data (flat, robust deviation, RANSAC correlation).
%   4  Two branches, each resampled, with bad channels interpolated (spherical splines) and
%      average-referenced:
%        ERP branch          cfg.hpERP high-pass (0.1 Hz)
%        ICA training branch cfg.hpICA - cfg.lpICA band-pass (2-100 Hz)
%   5  ERPLAB event list and bins (Binlister bin descriptor file or an existing event list).
%   6  OPTICAT-style ICA: extended infomax on stimulus-locked epochs, plus
%      overweighted saccade-onset segments when saccade detection is valid. Fixed random seed.
%   7  IC classification (ICLabel + EOG correlation + saccade/fixation variance ratio), with a
%      manual override file <subj>_ic_decisions.csv.
%   8  ICA weights applied to the 0.1-Hz branch; flagged ICs removed.
%   9  Low-pass, epoch, baseline-correct, flag artefacts (absolute threshold and moving-window
%      peak-to-peak) on EEG channels.
%   10 Linked-mastoid copy of the cleaned epochs (same ICA, same rejection flags), so the two
%      references differ only in the reference. Skipped if the mastoid channels are absent.
%   11 Trial-level table (optional saccade counts, optional single-trial ROI amplitudes) and
%      standardized measurement error (SME) per bin.
%   12 Averaging (accepted trials only) and saving.
%
% DESIGN NOTES
%   - Bad channels are detected AFTER re-referencing. Electrodes next to the online reference have
%     near-zero variance and weak correlations with their neighbours in online-referenced data and
%     get flagged wrongly; once re-referenced they look normal. A dead channel does the opposite
%     (it stops being flat after re-referencing), so the flatline test runs on the raw signal.
%   - Robust deviation (IQR-based) is used instead of kurtosis: kurtosis flags blink-heavy frontal
%     channels and channels with sporadic muscle bursts, which ICA and trial rejection handle.
%   - All channel bookkeeping uses channel LABELS, never indices (indices shift after removal).
%   - A saved ICA (<subj>_ica.mat) is reused only if the channel set AND bad-channel set match.
%   - rng(cfg.seed) is set per subject and runica 'rndreset' is off, so results are reproducible.
%   - Both branches share the same interpolation and referencing, so ICA weights from the
%     2-Hz branch transfer to the 0.1-Hz branch.
%
% OUTPUTS (under cfg.outputRoot)
%   epochs/              <subj>_preprocessed.set     cleaned epochs, average reference
%   erp_avgref/          <subj>_avg.erp              averaged ERP, average reference
%   erp_linkedmastoid/   <subj>_avg_lm.erp           averaged ERP, linked mastoids
%   qc_logs/             <subj>_qc.json              bad channels, ICs removed, trials per bin, flags
%   ic_review/           <subj>_ica.mat, _ic_metrics.csv, _IC_topos.png, optional _ic_decisions.csv
%   trial_level/         <subj>_trials.csv, <subj>_SME.csv
%
% QUICK START
%   1. Edit the USER SETTINGS block below (paths, file pattern, trigger codes, channel names).
%   2. Provide bins: a Binlister bin descriptor file (cfg.binDescriptorPath) or an existing
%      ERPLAB event list (cfg.eventlistPath).
%   3. Set cfg.subjectsToRun to one or two subjects first, check qc_logs and ic_review, then run all.
%
% ASSUMPTIONS
%   - Continuous data in microvolts, with an EEG channel montage that has (or can be given)
%     3-D coordinates (needed for interpolation, ICLabel and topographies; see cfg.chanlocFile).
%   - Epoch rejection assumes ERPs, not ongoing oscillations; adapt the thresholds for other designs.
%
% REQUIREMENTS
%   MATLAB (R2019b or newer; Signal Processing Toolbox; Parallel Computing Toolbox only if
%   cfg.nWorkers > 0), EEGLAB with ERPLAB, ICLabel and clean_rawdata plugins, plus the import
%   plugin for your file format (Curry, BrainVision, BIOSIG).
%   Developed with: MATLAB R2023b, EEGLAB 2023.1, ERPLAB 10.11, ICLabel 1.7.

clear; clc;
oldPool = gcp('nocreate');
if ~isempty(oldPool), delete(oldPool); end
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab; %#ok<ASGLU>
set(0, 'DefaultFigureVisible', 'off');

%% ======================== USER SETTINGS ========================
cfg = struct();

% ---- Paths and files -------------------------------------------------------------------
cfg.dataDir        = 'path/to/raw_data';      % searched recursively
cfg.outputRoot     = 'path/to/output';        % all outputs are written below this folder
cfg.filePattern    = '*.set';                 % e.g. '*.cdt', '*.vhdr', '*.bdf', '*.edf', '*.set' (char or cell array)
cfg.loader         = [];                      % optional custom loader: @(filepath) EEG; overrides the extension-based choice
cfg.subjectIdFrom  = 'filename';              % 'filename' | 'folder' (folder holding the file) | 'parentfolder'
cfg.subjectIdRegex = '';                      % optional regexp; the first match becomes the subject ID, e.g. 'sub-\d+'
cfg.subjectsToRun  = {};                      % {} = all subjects; e.g. {'sub-001','sub-002'} for a trial run
cfg.chanlocFile    = '';                      % optional channel-location file for pop_chanedit lookup, if the recordings lack coordinates

% ---- Events and bins -------------------------------------------------------------------
cfg.stimCodes        = [];                    % numeric codes of stimulus-onset events; [] = every numeric event
cfg.minStimEvents    = 10;                    % fewer stimulus events than this -> subject excluded
cfg.binDescriptorPath = '';                   % Binlister bin descriptor file (preferred) ...
cfg.eventlistPath    = '';                    % ... or an existing ERPLAB event list text file
cfg.minTrialsPerBinFlag = 15;                 % QC flag if a bin has fewer accepted trials

% ---- Channels --------------------------------------------------------------------------
cfg.eogChans      = {'VEOG', 'HEOG'};         % {vertical, horizontal} bipolar EOG; both needed for saccade detection and EOG
                                              % correlation. If either is missing, EOG channels are dropped and only ICLabel is used.
cfg.dropChans     = {'EKG', 'EMG', 'TRIGGER'};% non-EEG channels to discard (case-insensitive)
cfg.mastoidChans  = {'M1', 'M2'};             % for the linked-mastoid output; if absent, that output is skipped
cfg.linkedMastoidOutput = true;
cfg.roiChans      = {};                       % electrodes whose interpolation is flagged; {} = channels used in cfg.measures

% ---- Execution -------------------------------------------------------------------------
cfg.nWorkers = 0;        % 0 = serial loop; N > 0 = parpool with N workers (Parallel Computing Toolbox)
cfg.seed     = 1;
cfg.reuseICA = true;     % reuse <subj>_ica.mat if channels and bad channels are unchanged (manual-review loop)

% ---- Filtering and resampling ----------------------------------------------------------
cfg.fsNew    = 250;
cfg.hpERP    = 0.1;      % Hz, ERP branch
cfg.lpFinal  = 30;       % Hz, ERP branch, applied before epoching
cfg.hpICA    = 2;        % Hz, ICA training branch
cfg.lpICA    = 100;      % Hz, keeps the saccadic spike potential in the training data
cfg.hpDetect = 1;        % Hz, copy used for bad-channel detection

% ---- Bad channels ----------------------------------------------------------------------
cfg.flatRawSD       = 0.1;    % uV, raw SD below this = dead channel
cfg.flatSeconds     = 5;      % s of constant signal (clean_rawdata FlatlineCriterion)
cfg.devZThresh      = 5;      % robust z of IQR-based channel amplitude (PREP deviation criterion)
cfg.chanCorrThresh  = 0.8;    % RANSAC predicted-signal correlation (clean_rawdata default)
cfg.corrExemptChans = {'M1', 'M2'};  % edge electrodes with few neighbours: flat + deviation criteria only
cfg.maxBadChanProp  = 0.1;    % QC flag above this proportion of EEG channels
cfg.globalFailProp  = 0.5;    % proportion of flat channels -> subject excluded

% ---- Saccade detection (EOG, original sampling rate) -------------------------------------
cfg.sacLP         = 40;       % Hz
cfg.sacLambda     = 6;        % velocity threshold in median-based SDs
cfg.sacMinDurMs   = 12;
cfg.sacMergeMs    = 20;
cfg.sacMinAmpUV   = 10;
cfg.blinkAmpUV    = 150;
cfg.blinkMinDurMs = 50;
cfg.sacRateBounds = [0.2 5];  % saccades/s outside this range -> detection judged unreliable

% ---- OPTICAT / ICA ---------------------------------------------------------------------
cfg.icaEpochWindow = [-200 1000];   % ms; wider = more training data, but avoid heavy epoch overlap
cfg.icaPreRejAmp   = [-500 500];    % uV, gross-artefact pre-rejection of training epochs
cfg.owWindowMs     = [-20 10];      % overweighted saccade-onset segment (ms)
cfg.owProportion   = 0.5;
cfg.iclabelEyeThresh   = 0.7;       % ICLabel Eye probability
cfg.eogCorrThresh      = 0.4;       % |r| with EOG (0.1-Hz branch) ...
cfg.eogCorrBrainMax    = 0.5;       % ... counts only for ICs with ICLabel Brain below this ...
cfg.eogCorrStrong      = 0.6;       % ... unless |r| reaches this. EOG electrodes also pick up frontal EEG, so a
                                    % moderate correlation alone must not remove brain-like ICs.
cfg.varRatioStrong     = 3;         % saccade/fixation variance ratio ...
cfg.varRatioMaxBrain   = 0.3;       % ... combined with ICLabel Brain below this
cfg.maxRemoveProp      = 0.4;       % QC flag if more than this proportion of ICs is removed
cfg.iclabelOtherThresh = 0.9;
cfg.iclabelOtherClasses = [2 4 5 6];% ICLabel classes: Muscle, Heart, Line Noise, Channel Noise

% ---- Final epochs (relative to stimulus onset, ms) ----------------------------------------
cfg.erpEpochWindow = [-200 1000];
cfg.baselineWindow = [-200 0];
cfg.artWindow      = [-200 1000];
cfg.finalAmpThresh = [-100 100];    % uV, absolute threshold
cfg.movWinThresh   = 100;           % uV, moving-window peak-to-peak
cfg.movWinSize     = 200;           % ms
cfg.movWinStep     = 50;            % ms

% ---- Single-trial measures (EXAMPLE: edit for your paradigm, or set to [] to skip) -----------
% Mean amplitude over the listed channels and window, per trial and reference. Windows should come
% from the literature or a condition-blind collapsed localiser, not from the condition effects.
cfg.measures = struct( ...
    'name',  {'P1_occ',         'N1',                    'P300'}, ...
    'chans', {{'O1','OZ','O2'}, {'P7','P8','PO7','PO8'}, {'CZ','CPZ','PZ'}}, ...
    'win',   {[80 130],         [130 200],               [300 600]});
% Saccade/blink counts per trial in these windows (ms from onset); only used if EOG is available.
cfg.sacWindows = [-200 0; 0 100; 100 200; 200 400; 400 1000];
%% ======================== END USER SETTINGS ========================

cfg.version = 'eeg_preprocessing_pipeline';
if isempty(cfg.roiChans) && ~isempty(cfg.measures), cfg.roiChans = unique([cfg.measures.chans]); end
if ~isfolder(cfg.dataDir), error('cfg.dataDir not found: %s. Edit USER SETTINGS.', cfg.dataDir); end
if isempty(cfg.binDescriptorPath) && isempty(cfg.eventlistPath)
    error('Set cfg.binDescriptorPath (Binlister bin descriptor file) or cfg.eventlistPath (ERPLAB event list).');
end

cfg.outputDir      = fullfile(cfg.outputRoot, 'epochs');
cfg.outputDirERP   = fullfile(cfg.outputRoot, 'erp_avgref');
cfg.outputDirERPlm = fullfile(cfg.outputRoot, 'erp_linkedmastoid');
cfg.logDir         = fullfile(cfg.outputRoot, 'qc_logs');
cfg.reviewDir      = fullfile(cfg.outputRoot, 'ic_review');
cfg.trialDir       = fullfile(cfg.outputRoot, 'trial_level');

patterns = cfg.filePattern;
if ischar(patterns), patterns = {patterns}; end
files = [];
for p = 1:numel(patterns)
    f = dir(fullfile(cfg.dataDir, '**', patterns{p}));
    files = [files; f(~[f.isdir])]; %#ok<AGROW>
end
files = files(~startsWith({files.folder}, cfg.outputRoot));   % never re-read our own outputs
subjIDs = arrayfun(@(f) getSubjectID(f, cfg), files, 'UniformOutput', false);
if numel(unique(subjIDs)) ~= numel(subjIDs)
    error('Subject IDs are not unique. Adjust cfg.subjectIdFrom / cfg.subjectIdRegex.');
end
if ~isempty(cfg.subjectsToRun)
    keep = ismember(subjIDs, cfg.subjectsToRun);
    files = files(keep); subjIDs = subjIDs(keep);
end
if isempty(files), error('No files matching %s found in %s.', strjoin(patterns, ', '), cfg.dataDir); end

dirsToMake = {cfg.outputDir, cfg.outputDirERP, cfg.logDir, cfg.reviewDir, cfg.trialDir};
if cfg.linkedMastoidOutput, dirsToMake{end + 1} = cfg.outputDirERPlm; end
for d = dirsToMake
    if ~exist(d{1}, 'dir'), mkdir(d{1}); end
end

fprintf('Subjects to process: %d\n', numel(files));

%% MAIN LOOP
if cfg.nWorkers > 0
    nW = min(numel(files), cfg.nWorkers);
    poolObj = gcp('nocreate');
    if isempty(poolObj), poolObj = parpool(nW); end
    wait(parfevalOnAll(poolObj, @() set(0, 'DefaultFigureVisible', 'off'), 0));
    wait(parfevalOnAll(poolObj, @() maxNumCompThreads(1), 0));
    futures(numel(files)) = parallel.FevalFuture;
    for i = 1:numel(files)
        futures(i) = parfeval(poolObj, @processSubject, 0, files(i), subjIDs{i}, cfg);
    end
    for k = 1:numel(files)
        idx = fetchNext(futures);
        if ~isempty(futures(idx).Diary), fprintf('%s', futures(idx).Diary); end
        if ~isempty(futures(idx).Error)
            fprintf('ERROR dispatching %s: %s\n', files(idx).name, futures(idx).Error.message);
        end
        fprintf('--- (%d/%d finished) ---\n\n', k, numel(files));
    end
else
    for i = 1:numel(files)
        processSubject(files(i), subjIDs{i}, cfg);
        fprintf('--- (%d/%d finished) ---\n\n', i, numel(files));
    end
end

%% SERIAL PASS: IC review figures (topoplot is unreliable inside parallel workers)
icaFiles = dir(fullfile(cfg.reviewDir, '*_ica.mat'));
for i = 1:numel(icaFiles)
    try
        exportICReview(fullfile(cfg.reviewDir, icaFiles(i).name), cfg.reviewDir);
    catch figME
        fprintf('IC figure export failed for %s: %s\n', icaFiles(i).name, figME.message);
    end
end
fprintf('\nDone. QC logs: %s\nIC review: %s\nTrial-level: %s\n', cfg.logDir, cfg.reviewDir, cfg.trialDir);

%% ======================================================================
function processSubject(fileEntry, subj, cfg)
    fprintf('\n=== Processing %s ===\n', subj);
    rng(cfg.seed, 'twister');
    qc = struct('subject', subj, 'pipeline', cfg.version, 'excluded', false, 'exclusionReason', '', ...
                'badChannels', {{}}, 'badByFlat', {{}}, 'badByDeviation', {{}}, 'badByCorrelation', {{}}, ...
                'roiInterpolated', {{}}, 'mastoidInterpolated', {{}}, 'icaRank', NaN, 'icaMode', '', 'saccades', [], ...
                'icRemoved', [], 'icRemovedOcular', [], 'icRemovedOther', [], 'icDecisionSource', '', ...
                'binLabels', {{}}, 'epochsPerBin', [], 'trialsPerBin', [], 'flags', {{}});
    try
        %% Step 0: load recording, channel locations, stimulus events
        fpath = fullfile(fileEntry.folder, fileEntry.name);
        EEG_raw = loadRecording(fpath, cfg);
        if ~isempty(cfg.chanlocFile)
            EEG_raw = pop_chanedit(EEG_raw, 'lookup', cfg.chanlocFile);
        end
        [stimLatMs, stimCodes] = stimOnsetsMs(EEG_raw, cfg.stimCodes);
        if numel(stimLatMs) < cfg.minStimEvents
            error('Only %d stimulus events found (cfg.minStimEvents = %d).', numel(stimLatMs), cfg.minStimEvents);
        end

        %% Step 1: channel bookkeeping (keep bipolar EOG, drop other non-EEG)
        labels = upper({EEG_raw.chanlocs.labels});
        dropIdx = find(ismember(labels, upper(cfg.dropChans)));
        if ~isempty(dropIdx), EEG_raw = pop_select(EEG_raw, 'nochannel', dropIdx); end
        labels = upper({EEG_raw.chanlocs.labels});
        eogFound = nan(1, numel(cfg.eogChans));
        for k = 1:numel(cfg.eogChans)
            j = find(strcmp(labels, upper(cfg.eogChans{k})), 1);
            if ~isempty(j), eogFound(k) = j; end
        end
        hasEOG = numel(cfg.eogChans) == 2 && ~any(isnan(eogFound));
        if hasEOG
            eogIdxRaw = eogFound;
        else
            eogIdxRaw = [];
            qc.flags{end+1} = 'Vertical and horizontal EOG not both available: no saccade detection, no EOG correlation; ICLabel only.';
        end
        eegIdxRaw = setdiff(1:EEG_raw.nbchan, eogFound(~isnan(eogFound)));   % partial EOG channels are discarded
        if ~hasCoordinates(EEG_raw, eegIdxRaw)
            error('EEG channels without 3-D coordinates. Set cfg.chanlocFile or provide a montage in the recording.');
        end

        %% Step 2: saccade detection on raw EOG (original srate, absolute time in ms)
        taskSpan = [min(stimLatMs) + cfg.erpEpochWindow(1), max(stimLatMs) + cfg.erpEpochWindow(2)];
        if hasEOG
            sac = detectSaccades(double(EEG_raw.data(eogIdxRaw, :)), EEG_raw.srate, cfg);
            inTask = sac.onsetMs >= taskSpan(1) & sac.onsetMs <= taskSpan(2);
            sacRate = sum(inTask & ~sac.isBlink) / (diff(taskSpan) / 1000);
            sacOK = sacRate >= cfg.sacRateBounds(1) && sacRate <= cfg.sacRateBounds(2);
            qc.saccades = struct('nSaccades', sum(inTask & ~sac.isBlink), 'nBlinks', sum(inTask & sac.isBlink), ...
                                 'ratePerSec', sacRate, 'detectionOK', sacOK);
            if ~sacOK
                qc.flags{end+1} = sprintf('Saccade rate %.2f/s outside bounds: no OPTICAT overweighting; ocular ICs by ICLabel/EOG correlation only.', sacRate);
            end
        else
            sac = struct('onsetMs', [], 'offsetMs', [], 'durMs', [], 'ampUV', [], 'isBlink', false(1, 0), 'valid', false);
            sacRate = NaN; sacOK = false;
            qc.saccades = struct('nSaccades', NaN, 'nBlinks', NaN, 'ratePerSec', NaN, 'detectionOK', false);
        end

        %% Step 3: bad-channel detection on referenced data (labels throughout)
        EEG_eeg = pop_select(EEG_raw, 'channel', eegIdxRaw);
        bc = detectBadChannels(EEG_eeg, cfg);
        badLabels = bc.bad;
        qc.badChannels = badLabels; qc.badByFlat = bc.flat; qc.badByDeviation = bc.dev; qc.badByCorrelation = bc.corr;
        qc.roiInterpolated = intersect(upper(badLabels), upper(cfg.roiChans));
        qc.mastoidInterpolated = intersect(upper(badLabels), upper(cfg.mastoidChans));
        if numel(badLabels) > cfg.maxBadChanProp * numel(eegIdxRaw)
            qc.flags{end+1} = sprintf('%d bad channels (> %.0f%% of %d): inspect raw data before use.', ...
                numel(badLabels), 100 * cfg.maxBadChanProp, numel(eegIdxRaw));
        end
        if ~isempty(qc.roiInterpolated)
            qc.flags{end+1} = sprintf('ROI electrode(s) interpolated: %s.', strjoin(qc.roiInterpolated, ', '));
        end
        if ~isempty(qc.mastoidInterpolated)
            qc.flags{end+1} = sprintf('Mastoid(s) interpolated (%s): linked-mastoid output uses an interpolated reference.', strjoin(qc.mastoidInterpolated, ', '));
        end
        clear EEG_eeg

        %% Step 4: branches (EEG: filter, resample, interpolate, average reference; EOG: filter, resample)
        EEG_ica = buildBranch(EEG_raw, eegIdxRaw, eogIdxRaw, badLabels, cfg.hpICA, cfg.lpICA, cfg);
        EEG_erp = buildBranch(EEG_raw, eegIdxRaw, eogIdxRaw, badLabels, cfg.hpERP, [], cfg);
        clear EEG_raw
        eegIdx = find(~strcmp({EEG_erp.chanlocs.type}, 'EOG'));
        eogIdx = find(strcmp({EEG_erp.chanlocs.type}, 'EOG'));

        %% Step 5: ERPLAB event list and bins
        if ~isempty(cfg.binDescriptorPath)
            EEG_erp = pop_creabasiceventlist(EEG_erp, 'AlphanumericCleaning', 'on', 'BoundaryNumeric', {-99}, 'BoundaryString', {'boundary'});
            EEG_erp = pop_binlister(EEG_erp, 'BDF', cfg.binDescriptorPath, 'IndexEL', 1, 'SendEL2', 'EEG', 'Voutput', 'EEG');
        else
            EEG_erp = pop_editeventlist(EEG_erp, 'AlphanumericCleaning', 'on', 'BoundaryNumeric', {-99}, ...
                'BoundaryString', {'boundary'}, 'List', cfg.eventlistPath, 'SendEL2', 'EEG', 'UpdateEEG', 'code', 'Warning', 'on');
        end
        EEG_ica.event = EEG_erp.event; EEG_ica.urevent = EEG_erp.urevent; EEG_ica.EVENTLIST = EEG_erp.EVENTLIST;
        [nBins, binLabels, binEventCounts] = binInfo(EEG_erp);
        if nBins == 0, error('No events were assigned to any bin: check the bin descriptor / event list and cfg.stimCodes.'); end
        if any(binEventCounts == 0)
            qc.flags{end+1} = sprintf('Bin(s) with no events: %s. Check trigger coding before analysis.', mat2str(find(binEventCounts == 0)));
        end
        qc.binLabels = binLabels;

        %% Step 6: OPTICAT-style ICA (or reuse a saved decomposition with the same channel/bad-channel set)
        icaFile = fullfile(cfg.reviewDir, [subj '_ica.mat']);
        eegLabels = {EEG_ica.chanlocs(eegIdx).labels};
        reuse = false;
        if cfg.reuseICA && isfile(icaFile)
            S = load(icaFile, 'W', 'Sph', 'eegLabels', 'badLabels', 'icaRankSaved', 'icaModeSaved');
            reuse = isfield(S, 'badLabels') && isequal(S.eegLabels, eegLabels) && isequal(sort(upper(S.badLabels)), sort(upper(badLabels)));
            if ~reuse, qc.flags{end+1} = 'Saved ICA ignored (channel or bad-channel set changed): ICA recomputed.'; end
        end
        if reuse
            W = S.W; Sph = S.Sph; qc.icaRank = S.icaRankSaved;
            qc.icaMode = [regexprep(S.icaModeSaved, ' \(reused\)', '') ' (reused)'];
        else
            [W, Sph, qc.icaRank, qc.icaMode] = runOpticatICA(EEG_ica, eegIdx, numel(badLabels), sac, sacOK, cfg);
        end
        EEG_ica.icaweights = W; EEG_ica.icasphere = Sph; EEG_ica.icawinv = pinv(W * Sph);
        EEG_ica.icachansind = eegIdx; EEG_ica.icaact = [];
        EEG_ica = eeg_checkset(EEG_ica, 'ica');

        %% Step 7: IC classification
        EEG_ica = pop_iclabel(EEG_ica, 'default');
        P = EEG_ica.etc.ic_classification.ICLabel.classifications;
        nIC = size(W, 1);
        act = (W * Sph) * double(EEG_ica.data(eegIdx, :));
        vr = ocularMetrics(act, sac, EEG_ica.srate, taskSpan, sacOK);
        clear act
        if hasEOG
            % EOG correlation on the 0.1-Hz branch (corneoretinal signal is mostly < 2 Hz)
            actErp = (W * Sph) * double(EEG_erp.data(eegIdx, :));
            vIdx = eogIdx(strcmpi({EEG_erp.chanlocs(eogIdx).labels}, cfg.eogChans{1}));
            hIdx = eogIdx(strcmpi({EEG_erp.chanlocs(eogIdx).labels}, cfg.eogChans{2}));
            rV = rowCorr(actErp, double(EEG_erp.data(vIdx, :)));
            rH = rowCorr(actErp, double(EEG_erp.data(hIdx, :)));
            clear actErp
            eogR = max(abs(rV(:)), abs(rH(:)));
        else
            rV = nan(1, nIC); rH = nan(1, nIC); eogR = zeros(nIC, 1);
        end
        % EOG electrodes also record frontal EEG, and slow (< 2 Hz) brain activity correlates with them
        % on the 0.1-Hz branch: a moderate EOG correlation alone is not enough for a brain-like IC.
        eogRule = eogR >= cfg.eogCorrThresh & (P(:, 1) < cfg.eogCorrBrainMax | eogR >= cfg.eogCorrStrong);
        ocular = find(P(:, 3) >= cfg.iclabelEyeThresh | eogRule | ...
                      (sacOK & vr(:) >= cfg.varRatioStrong & P(:, 1) < cfg.varRatioMaxBrain))';
        other  = find(any(P(:, cfg.iclabelOtherClasses) >= cfg.iclabelOtherThresh, 2))';
        other  = setdiff(other, ocular);
        removeIC = union(ocular, other);
        qc.icDecisionSource = 'automatic';
        decFile = fullfile(cfg.reviewDir, [subj '_ic_decisions.csv']);   % columns: ic, remove (1/0)
        if isfile(decFile)
            D = readtable(decFile);
            removeIC = D.ic(D.remove == 1)';
            qc.icDecisionSource = 'manual (ic_decisions.csv)';
        end
        if any(removeIC < 1 | removeIC > nIC)
            error('IC indices to remove (%s) outside 1-%d.', mat2str(removeIC), nIC);
        end
        if numel(removeIC) > cfg.maxRemoveProp * nIC
            qc.flags{end+1} = sprintf('%d of %d ICs removed (> %.0f%%): review IC decisions.', numel(removeIC), nIC, 100 * cfg.maxRemoveProp);
        end
        qc.icRemoved = removeIC; qc.icRemovedOcular = intersect(ocular, removeIC); qc.icRemovedOther = intersect(other, removeIC);
        icn = (1:nIC)';
        M = array2table([icn, P, vr(:), rV(:), rH(:), ismember(icn, ocular), ismember(icn, other), ismember(icn, removeIC)], ...
            'VariableNames', {'ic', 'p_brain', 'p_muscle', 'p_eye', 'p_heart', 'p_line', 'p_channel', 'p_other', ...
                              'var_ratio_sacc_fix', 'r_eog_vertical', 'r_eog_horizontal', 'flag_ocular', 'flag_other', 'remove'});
        writetable(M, fullfile(cfg.reviewDir, [subj '_ic_metrics.csv']));
        chanlocsEEG = EEG_ica.chanlocs(eegIdx); %#ok<NASGU>
        icaRankSaved = qc.icaRank; icaModeSaved = qc.icaMode; icawinv = EEG_ica.icawinv; metrics = M; %#ok<NASGU>
        save(icaFile, 'W', 'Sph', 'icawinv', 'eegLabels', 'badLabels', 'chanlocsEEG', 'icaRankSaved', 'icaModeSaved', 'metrics', 'removeIC', 'subj');
        clear EEG_ica

        %% Step 8: apply ICA to the ERP branch (EEG channels only; EOG untouched)
        EEG_erp.icaweights = W; EEG_erp.icasphere = Sph; EEG_erp.icawinv = pinv(W * Sph);
        EEG_erp.icachansind = eegIdx; EEG_erp.icaact = [];
        EEG_erp = eeg_checkset(EEG_erp, 'ica');
        if ~isempty(removeIC), EEG_erp = pop_subcomp(EEG_erp, removeIC, 0); end

        %% Step 9: low-pass, epoch, baseline, artefact detection (EEG channels only)
        EEG_erp = pop_basicfilter(EEG_erp, 1:EEG_erp.nbchan, 'Boundary', 'boundary', 'Cutoff', cfg.lpFinal, ...
            'Design', 'butter', 'Filter', 'lowpass', 'Order', 2);
        EEG_erp = pop_epochbin(EEG_erp, cfg.erpEpochWindow, 'none');
        EEG_erp = pop_rmbase(EEG_erp, cfg.baselineWindow);
        EEG_erp = pop_artextval(EEG_erp, 'Channel', eegIdx, 'Flag', [1 4], 'Threshold', cfg.finalAmpThresh, 'Twindow', cfg.artWindow);
        EEG_erp = pop_artmwppth(EEG_erp, 'Channel', eegIdx, 'Flag', 2, 'Threshold', cfg.movWinThresh, ...
            'Twindow', cfg.artWindow, 'Windowsize', cfg.movWinSize, 'Windowstep', cfg.movWinStep);

        %% Step 10: linked-mastoid copy (same data, same rejection flags; M1/M2 rows become +/-(M1-M2)/2)
        EEG_lm = [];
        if cfg.linkedMastoidOutput
            mIdx = eegIdx(ismember(upper({EEG_erp.chanlocs(eegIdx).labels}), upper(cfg.mastoidChans)));
            if numel(cfg.mastoidChans) == 2 && numel(mIdx) == 2
                EEG_lm = EEG_erp;
                EEG_lm.data(eegIdx, :, :) = EEG_lm.data(eegIdx, :, :) - mean(EEG_lm.data(mIdx, :, :), 1);
                EEG_lm.ref = 'linked mastoids (derived from average-referenced, ICA-cleaned epochs)';
            else
                qc.flags{end+1} = 'Two mastoid channels not found: linked-mastoid output skipped.';
            end
        end

        %% Step 11: trial-level table and SME
        [T, smeT, binCounts, epochCounts] = trialLevelOutputs(EEG_erp, EEG_lm, sac, stimLatMs, stimCodes, subj, cfg);
        writetable(T, fullfile(cfg.trialDir, [subj '_trials.csv']));
        writetable(smeT, fullfile(cfg.trialDir, [subj '_SME.csv']));
        qc.trialsPerBin = binCounts; qc.epochsPerBin = epochCounts;
        if any(binCounts < cfg.minTrialsPerBinFlag)
            qc.flags{end+1} = sprintf('Bin(s) below %d trials: %s', cfg.minTrialsPerBinFlag, mat2str(find(binCounts < cfg.minTrialsPerBinFlag)));
        end

        %% Step 12: average and save
        ERP = pop_averager(EEG_erp, 'Criterion', 'good', 'ExcludeBoundary', 'on', 'SEM', 'on');
        ERP.ref = 'average';
        pop_savemyerp(ERP, 'erpname', [subj '_avg'], 'filename', [subj '_avg.erp'], 'filepath', cfg.outputDirERP, 'Warning', 'off');
        if ~isempty(EEG_lm)
            ERPlm = pop_averager(EEG_lm, 'Criterion', 'good', 'ExcludeBoundary', 'on', 'SEM', 'on');
            ERPlm.ref = 'linked mastoids';
            pop_savemyerp(ERPlm, 'erpname', [subj '_avg_lm'], 'filename', [subj '_avg_lm.erp'], 'filepath', cfg.outputDirERPlm, 'Warning', 'off');
        end
        EEG_erp.ref = 'average';
        pop_saveset(EEG_erp, 'filename', [subj '_preprocessed.set'], 'filepath', cfg.outputDir);
        writeQC(cfg.logDir, qc);
        fprintf('Finished %s: %d bad chans [%s], saccades %s, ICA %s, %d ICs removed (%d ocular), trials/bin %s\n', ...
            subj, numel(badLabels), strjoin(badLabels, ' '), ...
            iff(hasEOG, sprintf('%d (%.2f/s)', qc.saccades.nSaccades, sacRate), 'n/a'), qc.icaMode, ...
            numel(removeIC), numel(qc.icRemovedOcular), mat2str(binCounts));
    catch ME
        qc.excluded = true;
        if isempty(ME.stack), where = ''; else, where = sprintf(' (line %d in %s)', ME.stack(1).line, ME.stack(1).name); end
        qc.exclusionReason = sprintf('Processing error: %s%s', ME.message, where);
        writeQC(cfg.logDir, qc);
        fprintf('ERROR %s: %s\n', subj, qc.exclusionReason);
    end
end

%% ======================================================================
function subj = getSubjectID(fileEntry, cfg)
% Subject ID from the file name or folder structure; made safe for use in file names.
    [~, base] = fileparts(fileEntry.name);
    switch lower(cfg.subjectIdFrom)
        case 'filename',     subj = base;
        case 'folder',       [~, subj] = fileparts(fileEntry.folder);
        case 'parentfolder', [~, subj] = fileparts(fileparts(fileEntry.folder));
        otherwise, error('cfg.subjectIdFrom must be ''filename'', ''folder'' or ''parentfolder''.');
    end
    if ~isempty(cfg.subjectIdRegex)
        m = regexp(subj, cfg.subjectIdRegex, 'match', 'once');
        if ~isempty(m), subj = m; end
    end
    subj = regexprep(subj, '[^\w\-]', '_');
end

function EEG = loadRecording(fpath, cfg)
% Choose the importer by file extension (or use cfg.loader). Add cases for other formats.
    [folder, name, ext] = fileparts(fpath);
    if ~isempty(cfg.loader), EEG = cfg.loader(fpath); return; end
    switch lower(ext)
        case '.cdt'
            if ~isfile([fpath '.dpo']), error('Curry parameter file not found: %s.dpo', fpath); end
            EEG = pop_loadcurry(fpath);
        case '.vhdr'
            EEG = pop_loadbv(folder, [name ext]);
        case {'.bdf', '.edf'}
            EEG = pop_biosig(fpath);
        case '.set'
            EEG = pop_loadset('filename', [name ext], 'filepath', folder);
        otherwise
            error('No loader for extension %s: set cfg.loader.', ext);
    end
end

function ok = hasCoordinates(EEG, chanIdx)
    ok = ~isempty(EEG.chanlocs) && isfield(EEG.chanlocs, 'X') && ...
         ~any(arrayfun(@(k) isempty(EEG.chanlocs(k).X), chanIdx));
end

function out = iff(cond, a, b)
    if cond, out = a; else, out = b; end
end

%% ======================================================================
function bc = detectBadChannels(EEG_eeg, cfg)
% Bad-channel detection that is independent of the online reference.
%   Flat:  raw SD < cfg.flatRawSD, or >= cfg.flatSeconds of constant signal (online-referenced data;
%          a dead channel stops being flat once re-referenced, so this test must come first).
%   Pass 1: 1-Hz HP copy re-referenced to the per-sample median of the non-flat channels;
%          robust deviation (|z| > cfg.devZThresh) and RANSAC correlation (< cfg.chanCorrThresh).
%   Pass 2: same tests on the remaining channels, re-referenced to their average.
% All results are returned as channel labels.
    labs = {EEG_eeg.chanlocs.labels};
    nCh = numel(labs);
    rawSD = std(double(EEG_eeg.data), 0, 2)';
    flat = labs(rawSD < cfg.flatRawSD);
    E = pop_basicfilter(EEG_eeg, 1:nCh, 'Boundary', 'boundary', 'Cutoff', cfg.hpDetect, ...
        'Design', 'butter', 'Filter', 'highpass', 'Order', 2, 'RemoveDC', 'on');
    E = pop_resample(E, cfg.fsNew);
    try
        Ef = pop_clean_rawdata(E, 'FlatlineCriterion', cfg.flatSeconds, 'ChannelCriterion', 'off', ...
            'LineNoiseCriterion', 'off', 'Highpass', 'off', 'BurstCriterion', 'off', 'WindowCriterion', 'off');
        flat = union(flat, setdiff(labs, {Ef.chanlocs.labels}));
        clear Ef
    catch
        % FlatlineCriterion unavailable: raw-SD test only
    end
    if numel(flat) / nCh >= cfg.globalFailProp
        error('%d of %d channels flat (global recording failure).', numel(flat), nCh);
    end
    dev = {}; cor = {};
    bad = flat;
    for pass = 1:2
        good = setdiff(labs, bad, 'stable');
        Eg = pop_select(E, 'channel', good);
        X = double(Eg.data);
        if pass == 1, X = X - median(X, 1); else, X = X - mean(X, 1); end
        Eg.data = single(X);
        % Robust deviation (PREP-style): channel amplitude = 0.7413 * IQR,
        % robust z across channels. IQR ignores sparse blinks/EMG bursts, so frontal blink channels
        % and channels with occasional temporalis bursts are not flagged (kurtosis flags both).
        amp = 0.7413 * rowIQR(X);
        dz = (amp - median(amp)) / (1.4826 * median(abs(amp - median(amp))) + eps);
        dNew = {Eg.chanlocs(abs(dz) > cfg.devZThresh).labels};
        cNew = {};
        try
            rng(cfg.seed, 'twister');
            Ec = pop_clean_rawdata(Eg, 'FlatlineCriterion', 'off', 'ChannelCriterion', cfg.chanCorrThresh, ...
                'LineNoiseCriterion', 'off', 'Highpass', 'off', 'BurstCriterion', 'off', 'WindowCriterion', 'off');
            cNew = setdiff({Eg.chanlocs.labels}, {Ec.chanlocs.labels});
            % edge electrodes have few neighbours and naturally lower RANSAC correlation:
            % they are judged by the flat and deviation criteria only
            cNew = cNew(~ismember(upper(cNew), upper(cfg.corrExemptChans)));
            clear Ec
        catch
            if pass == 1, warning('clean_rawdata unavailable: deviation-only bad-channel detection.'); end
        end
        dev = union(dev, dNew); cor = union(cor, cNew);
        bad = union(bad, union(dNew, cNew));
        if isempty(dNew) && isempty(cNew), break; end
    end
    order = @(c) labs(ismember(labs, c));          % keep montage order
    bc = struct('bad', {order(bad)}, 'flat', {order(flat)}, 'dev', {order(dev)}, 'corr', {order(cor)});
end

function q = rowIQR(X)
% Interquartile range of each row (no Statistics Toolbox needed).
    S = sort(X, 2); n = size(S, 2);
    lo = S(:, max(1, round(0.25 * n))); hi = S(:, max(1, round(0.75 * n)));
    q = (hi - lo)';
end

%% ======================================================================
function EEG_out = buildBranch(EEG_raw, eegIdx, eogIdx, badLabels, hp, lp, cfg)
% EEG channels: high-pass (+ optional low-pass), resample, interpolate bad channels, average reference.
% EOG channels (if any): identical filtering and resampling only; appended with chanlocs.type = 'EOG'.
    hasEOG = ~isempty(eogIdx);
    E = pop_select(EEG_raw, 'channel', eegIdx);
    E = pop_basicfilter(E, 1:E.nbchan, 'Boundary', 'boundary', 'Cutoff', hp, 'Design', 'butter', 'Filter', 'highpass', 'Order', 2, 'RemoveDC', 'on');
    if ~isempty(lp)
        E = pop_basicfilter(E, 1:E.nbchan, 'Boundary', 'boundary', 'Cutoff', lp, 'Design', 'butter', 'Filter', 'lowpass', 'Order', 2);
    end
    E = pop_resample(E, cfg.fsNew);
    if hasEOG
        O = pop_select(EEG_raw, 'channel', eogIdx);
        O = pop_basicfilter(O, 1:O.nbchan, 'Boundary', 'boundary', 'Cutoff', hp, 'Design', 'butter', 'Filter', 'highpass', 'Order', 2, 'RemoveDC', 'on');
        if ~isempty(lp)
            O = pop_basicfilter(O, 1:O.nbchan, 'Boundary', 'boundary', 'Cutoff', lp, 'Design', 'butter', 'Filter', 'lowpass', 'Order', 2);
        end
        O = pop_resample(O, cfg.fsNew);
    end
    if ~isempty(badLabels)
        fullLocs = E.chanlocs;
        E = pop_select(E, 'nochannel', badLabels);
        E = pop_interp(E, fullLocs, 'spherical');
        if ~isequal({E.chanlocs.labels}, {fullLocs.labels})
            error('Channel order changed after interpolation.');
        end
    end
    E = pop_reref(E, []);          % average of all EEG channels (EOG is not in E)
    EEG_out = E;
    if hasEOG
        n = min(E.pnts, O.pnts);
        EEG_out.data = [E.data(:, 1:n); O.data(:, 1:n)];
        EEG_out.pnts = n;
        EEG_out.xmax = EEG_out.xmin + (n - 1) / EEG_out.srate;
        EEG_out.times = [];
        for k = 1:O.nbchan
            loc = E.chanlocs(1);
            fn = fieldnames(loc);
            for f = 1:numel(fn), loc.(fn{f}) = []; end
            loc.labels = O.chanlocs(k).labels; loc.type = 'EOG';
            EEG_out.chanlocs(end + 1) = loc;
        end
    end
    [EEG_out.chanlocs(1:E.nbchan).type] = deal('EEG');
    EEG_out.nbchan = size(EEG_out.data, 1);
    EEG_out.icaweights = []; EEG_out.icasphere = []; EEG_out.icawinv = []; EEG_out.icachansind = []; EEG_out.icaact = [];
    EEG_out = eeg_checkset(EEG_out);
end

%% ======================================================================
function [W, Sph, icaRank, icaMode] = runOpticatICA(EEG_ica, eegIdx, nInterp, sac, sacOK, cfg)
% Training data: stimulus-locked epochs (pre-rejected at cfg.icaPreRejAmp) plus, if saccade detection
% is valid, replicated saccade-onset segments (overweighting proportion cfg.owProportion).
% Rank = channels - interpolated - 1 (average reference), capped by the numerical rank.
    nEEG = numel(eegIdx);
    Ep = pop_epochbin(EEG_ica, cfg.icaEpochWindow, 'none');
    Ep = pop_artextval(Ep, 'Channel', eegIdx, 'Flag', 1, 'Threshold', cfg.icaPreRejAmp, 'Twindow', cfg.icaEpochWindow);
    good = ~Ep.reject.rejmanual;
    X = reshape(double(Ep.data(eegIdx, :, good)), nEEG, []);
    icaMode = 'epoched';
    if size(X, 2) < 20 * nEEG^2
        X = double(EEG_ica.data(eegIdx, :));
        icaMode = 'continuous (insufficient epoched data)';
    end
    if sacOK
        on = round(sac.onsetMs(~sac.isBlink) / 1000 * EEG_ica.srate) + 1;
        w = round(cfg.owWindowMs / 1000 * EEG_ica.srate);
        on = on(on + w(1) >= 1 & on + w(2) <= EEG_ica.pnts);
        if ~isempty(on)
            seg = cell(1, numel(on));
            for k = 1:numel(on), seg{k} = double(EEG_ica.data(eegIdx, on(k) + w(1):on(k) + w(2))); end
            segData = [seg{:}];
            nNeeded = round(cfg.owProportion * size(X, 2));
            reps = ceil(nNeeded / size(segData, 2));
            segData = repmat(segData, 1, reps);
            X = [X, segData(:, 1:nNeeded)];
            icaMode = [icaMode ' + OPTICAT overweighting'];
        end
    end
    X = X - mean(X, 2);
    ev = sort(eig(cov(X')), 'descend');
    icaRank = min(nEEG - 1 - nInterp, sum(ev > ev(1) * 1e-7));
    rng(cfg.seed, 'twister');
    try
        [W, Sph] = runica(X, 'extended', 1, 'pca', icaRank, 'interrupt', 'off', 'verbose', 'off', 'rndreset', 'no');
    catch   % older runica without the 'rndreset' option (seed already fixed by rng)
        rng(cfg.seed, 'twister');
        [W, Sph] = runica(X, 'extended', 1, 'pca', icaRank, 'interrupt', 'off', 'verbose', 'off');
    end
end

%% ======================================================================
function sac = detectSaccades(eog, srate, cfg)
% eog rows: vertical, horizontal (bipolar). Velocity-based detection with a 2-D elliptic threshold.
    [b, a] = butter(2, cfg.sacLP / (srate / 2), 'low');
    v = filtfilt(b, a, eog(1, :)); h = filtfilt(b, a, eog(2, :));
    vel = @(x) [0 0, (x(5:end) + x(4:end-1) - x(2:end-3) - x(1:end-4)) * srate / 6, 0 0];
    vv = vel(v); vh = vel(h);
    sd = @(x) sqrt(median(x.^2) - median(x)^2);
    thrV = cfg.sacLambda * sd(vv); thrH = cfg.sacLambda * sd(vh);
    test = (vv / thrV).^2 + (vh / thrH).^2 > 1;
    d = diff([0 test 0]); on = find(d == 1); off = find(d == -1) - 1;
    mergeS = round(cfg.sacMergeMs / 1000 * srate);
    k = 1;
    while k < numel(on)
        if on(k + 1) - off(k) <= mergeS
            off(k) = off(k + 1); on(k + 1) = []; off(k + 1) = [];
        else
            k = k + 1;
        end
    end
    dur = (off - on + 1) / srate * 1000;
    amp = zeros(size(on));
    for k = 1:numel(on)
        amp(k) = hypot(max(h(on(k):off(k))) - min(h(on(k):off(k))), max(v(on(k):off(k))) - min(v(on(k):off(k))));
    end
    keep = dur >= cfg.sacMinDurMs & amp >= cfg.sacMinAmpUV;
    on = on(keep); off = off(keep); dur = dur(keep); amp = amp(keep);
    isBlink = false(size(on));
    pad = round(0.05 * srate);
    for k = 1:numel(on)
        seg = v(max(1, on(k) - pad):min(numel(v), off(k) + pad));
        isBlink(k) = (max(seg) - min(seg)) > cfg.blinkAmpUV && dur(k) >= cfg.blinkMinDurMs;
    end
    sac = struct('onsetMs', (on - 1) / srate * 1000, 'offsetMs', (off - 1) / srate * 1000, ...
                 'durMs', dur, 'ampUV', amp, 'isBlink', isBlink, 'valid', true);
end

%% ======================================================================
function vr = ocularMetrics(act, sac, srate, taskSpan, sacOK)
% Saccade/fixation variance ratio per IC; NaN if saccade detection failed.
    nIC = size(act, 1); vr = nan(1, nIC);
    if ~sacOK, return; end
    toS = @(ms) max(1, min(size(act, 2), round(ms / 1000 * srate) + 1));
    sacMask = false(1, size(act, 2)); blinkMask = false(1, size(act, 2));
    for k = 1:numel(sac.onsetMs)
        if sac.isBlink(k)
            blinkMask(toS(sac.onsetMs(k) - 100):toS(sac.offsetMs(k) + 100)) = true;
        else
            sacMask(toS(sac.onsetMs(k)):toS(sac.offsetMs(k))) = true;
        end
    end
    taskMask = false(1, size(act, 2)); taskMask(toS(taskSpan(1)):toS(taskSpan(2))) = true;
    fixMask = taskMask & ~sacMask & ~blinkMask;
    sacMask = sacMask & taskMask & ~blinkMask;
    vr = var(act(:, sacMask), 0, 2)' ./ var(act(:, fixMask), 0, 2)';
end

function r = rowCorr(A, x)
% Pearson correlation of each row of A with vector x.
    A = A - mean(A, 2); x = x - mean(x);
    r = (A * x') ./ (sqrt(sum(A.^2, 2)) * sqrt(sum(x.^2)));
    r = r';
end

%% ======================================================================
function idx = stimEventIdx(EEG, stimCodes)
% Indices of stimulus events in chronological order. stimCodes = [] selects every numeric event.
    typ = eventCodes(EEG);
    if isempty(stimCodes), sel = ~isnan(typ); else, sel = ismember(typ, stimCodes); end
    idx = find(sel);
    [~, ord] = sort([EEG.event(idx).latency]); idx = idx(ord);
end

function [latMs, codes] = stimOnsetsMs(EEG, stimCodes)
    idx = stimEventIdx(EEG, stimCodes);
    typ = eventCodes(EEG);
    latMs = ([EEG.event(idx).latency] - 1) / EEG.srate * 1000;
    codes = typ(idx);
end

function typ = eventCodes(EEG)
% Numeric event codes. Strings are reduced to their digits (e.g. 'S 11' -> 11), mirroring the
% ERPLAB 'AlphanumericCleaning' option; non-numeric events such as 'boundary' become NaN.
    typ = nan(1, numel(EEG.event));
    for k = 1:numel(EEG.event)
        t = EEG.event(k).type;
        if ischar(t) || isstring(t), typ(k) = str2double(regexprep(char(t), '[^0-9]', '')); else, typ(k) = double(t); end
    end
end

%% ======================================================================
function [nBins, binLabels, evCounts] = binInfo(EEG)
% Number of bins, their labels and the number of events per bin, from the ERPLAB EVENTLIST.
    EL = EEG.EVENTLIST; ei = EL.eventinfo;
    allBins = [];
    for k = 1:numel(ei), allBins = [allBins, ei(k).bini(:)']; end %#ok<AGROW>
    allBins = allBins(allBins >= 1);
    if isfield(EL, 'nbin') && ~isempty(EL.nbin), nBins = EL.nbin; elseif isempty(allBins), nBins = 0; else, nBins = max(allBins); end
    binLabels = arrayfun(@(b) sprintf('bin%d', b), 1:nBins, 'UniformOutput', false);
    for b = 1:nBins
        try
            if ~isempty(EL.bdf(b).description), binLabels{b} = EL.bdf(b).description; end
        catch
            k = find(arrayfun(@(e) ~isempty(e.bini) && e.bini(1) == b && ~isempty(e.binlabel), ei), 1);
            if ~isempty(k), binLabels{b} = ei(k).binlabel; end
        end
    end
    evCounts = arrayfun(@(b) sum(allBins == b), 1:nBins);
end

%% ======================================================================
function [T, smeT, binCounts, epochCounts] = trialLevelOutputs(EEG, EEGlm, sac, stimLatMs, stimCodes, subj, cfg)
% One row per epoch: trial index, event code, bin, rejection flag, optional saccade/blink counts per
% window, optional single-trial ROI mean amplitudes (average reference; *_lm = linked mastoids).
% SME per bin and measure = SD / sqrt(n accepted).
    nE = EEG.trials; t = EEG.times;
    labs = upper({EEG.chanlocs.labels});
    hasLM = ~isempty(EEGlm);
    [nBins, binLabels] = binInfo(EEG);
    rows = cell(nE, 1);
    rej = EEG.reject.rejmanual(:)';
    if isempty(rej), rej = false(1, nE); end
    sacOn = sac.onsetMs(~sac.isBlink); blkOn = sac.onsetMs(sac.isBlink);
    [epochTrial, epochBin] = epochTrialIndex(EEG, stimLatMs);
    tEnd = cfg.erpEpochWindow(2);
    for e = 1:nE
        trial = epochTrial(e);
        onsetMs = stimLatMs(trial);
        r = struct('Subject', subj, 'epoch', e, 'trial', trial, 'eventCode', stimCodes(trial), ...
                   'bin', epochBin(e), 'condition', binLabels{epochBin(e)}, ...
                   'rejected', rej(e));
        if sac.valid
            for w = 1:size(cfg.sacWindows, 1)
                a = cfg.sacWindows(w, 1); b = cfg.sacWindows(w, 2);
                r.(sprintf('nSacc_%s_%s', numlab(a), numlab(b))) = sum(sacOn >= onsetMs + a & sacOn < onsetMs + b);
            end
            first = sacOn(sacOn >= onsetMs & sacOn < onsetMs + tEnd);
            if isempty(first), r.firstSaccLatMs = NaN; else, r.firstSaccLatMs = min(first) - onsetMs; end
            r.(sprintf('nBlink_0_%d', tEnd)) = sum(blkOn >= onsetMs & blkOn < onsetMs + tEnd);
        end
        for m = 1:numel(cfg.measures)
            ci = find(ismember(labs, upper(cfg.measures(m).chans)));
            ti = t >= cfg.measures(m).win(1) & t < cfg.measures(m).win(2);
            r.(cfg.measures(m).name) = mean(mean(EEG.data(ci, ti, e), 1), 2);
            if hasLM, r.([cfg.measures(m).name '_lm']) = mean(mean(EEGlm.data(ci, ti, e), 1), 2); end
        end
        rows{e} = r;
    end
    T = struct2table([rows{:}]');
    T = sortrows(T, 'trial');
    binCounts   = arrayfun(@(b) sum(T.bin == b & ~T.rejected), 1:nBins);
    epochCounts = arrayfun(@(b) sum(T.bin == b), 1:nBins);
    refSfxs = {''};
    if hasLM, refSfxs = {'', '_lm'}; end
    s = {};
    for m = 1:numel(cfg.measures)
        for refSfx = refSfxs
            nm = [cfg.measures(m).name refSfx{1}];
            for b = 1:nBins
                x = T.(nm)(T.bin == b & ~T.rejected);
                s(end + 1, :) = {subj, nm, b, binLabels{b}, numel(x), mean(x), std(x), std(x) / sqrt(numel(x))}; %#ok<AGROW>
            end
        end
    end
    if isempty(s), s = cell(0, 8); end
    smeT = cell2table(s, 'VariableNames', {'Subject', 'measure', 'bin', 'condition', 'n', 'mean_uV', 'sd_uV', 'SME_uV'});
end

function [trialIdx, binIdx] = epochTrialIndex(EEG, stimLatMs)
% Chronological trial index of each epoch and its bin, matched through EVENTLIST.eventinfo
% (ERPLAB may rewrite event codes after binning, so codes cannot be used for matching).
    nE = EEG.trials; trialIdx = nan(1, nE); binIdx = nan(1, nE);
    ei = EEG.EVENTLIST.eventinfo;
    for k = 1:numel(ei)
        be = ei(k).bepoch; b = ei(k).bini;
        if isempty(be) || be < 1 || be > nE || isempty(b) || b(1) < 1, continue; end
        [d, j] = min(abs(stimLatMs - ei(k).time * 1000));
        if d <= 10, trialIdx(be) = j; binIdx(be) = b(1); end
    end
    if any(isnan(trialIdx)) || numel(unique(trialIdx)) ~= nE
        error(['Epoch-to-trial matching failed (%d unmatched, %d duplicates). ' ...
               'cfg.stimCodes must include every code that is binned.'], ...
               sum(isnan(trialIdx)), nE - numel(unique(trialIdx(~isnan(trialIdx)))));
    end
end

function s = numlab(x)
    if x < 0, s = sprintf('m%d', -x); else, s = sprintf('%d', x); end
end

%% ======================================================================
function exportICReview(icaFile, outDir)
% Topographies of all ICs, titled with ICLabel top class, variance ratio and removal decision.
    S = load(icaFile);
    nIC = size(S.icawinv, 2); nG = ceil(sqrt(nIC));
    classes = {'Brain', 'Muscle', 'Eye', 'Heart', 'Line', 'Chan', 'Other'};
    P = S.metrics{:, {'p_brain', 'p_muscle', 'p_eye', 'p_heart', 'p_line', 'p_channel', 'p_other'}};
    fh = figure('Visible', 'off', 'Position', [20 20 1800 1500]);
    for k = 1:nIC
        subplot(nG, nG, k);
        topoplot(S.icawinv(:, k), S.chanlocsEEG, 'electrodes', 'off');
        [pm, cm] = max(P(k, :));
        tag = ''; if S.metrics.remove(k), tag = ' [REMOVE]'; end
        title(sprintf('IC%d %s %.2f VR%.2f%s', k, classes{cm}, pm, S.metrics.var_ratio_sacc_fix(k), tag), 'FontSize', 7);
    end
    saveas(fh, fullfile(outDir, [S.subj '_IC_topos.png'])); close(fh);
end

function writeQC(logDir, qc)
    fid = fopen(fullfile(logDir, [qc.subject '_qc.json']), 'w');
    fprintf(fid, '%s', jsonencode(qc));
    fclose(fid);
end
