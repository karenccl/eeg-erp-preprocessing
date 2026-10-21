%% EEG/ERP PREPROCESSING PIPELINE (EEGLAB + ERPLAB)
%
% Dual-branch ICA pipeline for event-related EEG data. ICA is trained on a
% 1 Hz high-passed copy of the data (better decomposition) and the solution
% is applied to a 0.1 Hz high-passed copy (preserves slow ERP components).
%
% PIPELINE
%   1.  Load raw data (format inferred from file extension)
%   2.  Drop non-EEG channels; make sure channel locations exist
%   3.  Sanity checks: trigger count, global flat-lining
%   4.  Bad-channel detection on a 1 Hz high-passed copy
%       (flat-line + kurtosis + correlation via clean_rawdata)
%   5.  Build ICA branch (1 Hz HP) and ERP branch (0.1 Hz HP):
%       filter -> resample -> spherical interpolation -> average reference
%   6.  ERPLAB EventList + bin assignment (bin descriptor file)
%   7.  Data-rank estimation (empirical, cross-checked against theory)
%   8.  Epoch ICA branch, drop extreme epochs, run ICA with PCA rank
%       reduction (falls back to continuous data if too few samples)
%   9.  ICLabel; export topographies/spectra of flagged ICs for review;
%       auto-remove high-confidence artifact ICs
%   10. Transfer ICA solution to ERP branch, remove ICs, low-pass, epoch,
%       baseline-correct
%   11. Artifact detection (absolute threshold + moving-window peak-to-peak)
%   12. Average; save .set/.erp; per-subject QC JSON + batch QC summary CSV
%
% REQUIREMENTS
%   MATLAB R2021a+, EEGLAB 2022+, ERPLAB 9+, ICLabel plugin.
%   Optional: clean_rawdata (correlation-based bad-channel detection),
%   Picard plugin (cfg.icaType = 'picard'), Parallel Computing Toolbox.
%
% USAGE
%   1. Edit the CONFIG section below (paths, file pattern, epoch windows).
%   2. Write an ERPLAB bin descriptor file (BDF), e.g.:
%          bin 1
%          Condition A
%          .{11}
%
%          bin 2
%          Condition B
%          .{12}
%   3. Run the script. Outputs are written to cfg.outputDir:
%          sets/  epoched, artifact-marked .set files
%          erp/   averaged .erp files
%          qc/    per-subject QC JSON, IC review figures,
%                 batch_qc_summary.csv, config_used.json
%
% NOTES
%   - Default thresholds are common starting points, not universal truths.
%     Inspect the QC output and the IC review figures before trusting
%     automatic decisions.
%   - If your system stores the online reference electrode as an all-zero
%     channel, list it in cfg.nonEEGChans (or add it back with pop_reref's
%     'refloc' option) so it is not treated as a bad channel.
%
% Author:  <Your Name>
% License: MIT

clear; clc;

%% ======================= CONFIG (edit this) =============================
cfg = struct();

% ---- Input / output -----------------------------------------------------
cfg.dataDir           = fullfile(pwd, 'data', 'raw');
cfg.filePattern       = '*.set';      % e.g. '*.cdt', '*.vhdr', '*.bdf', '*.edf'
cfg.recursive         = true;         % search sub-folders of dataDir
cfg.outputDir         = fullfile(pwd, 'derivatives', 'preprocessed');
cfg.binDescriptorFile = fullfile(pwd, 'config', 'bins.txt');
cfg.chanlocLookup     = '';           % e.g. 'standard_1005.elc'; '' = use file's locations
cfg.subjectIdRegex    = '';           % e.g. 'sub-\d+'; '' = use file name
cfg.overwrite         = false;        % skip subjects whose output already exists

% ---- Channels -----------------------------------------------------------
cfg.nonEEGChans = {'VEOG','HEOG','EOG','EKG','ECG','EMG','TRIGGER'};  % removed if present

% ---- Sanity checks ------------------------------------------------------
cfg.minEventCount   = [];    % [] = skip; otherwise exclude if fewer events
cfg.minTrialsPerBin = 15;    % flag (not exclude) bins below this

% ---- Filtering / resampling ---------------------------------------------
cfg.hpICA       = 1;      % Hz, ICA-training branch
cfg.hpERP       = 0.1;    % Hz, ERP branch
cfg.lpERP       = 30;     % Hz, applied to continuous ERP data before epoching
cfg.filterOrder = 2;      % Butterworth order (ERPLAB convention)
cfg.fsNew       = 250;    % Hz

% ---- Bad-channel detection ----------------------------------------------
cfg.flatStdThresh   = 1;      % uV; channel SD below this = flat
cfg.maxPropFlat     = 0.5;    % exclude subject if >= this proportion flat
cfg.kurtosisThresh  = 5;      % z-scored kurtosis
cfg.chanCorrThresh  = 0.85;   % clean_rawdata ChannelCriterion
cfg.maxPropBadChans = 0.25;   % exclude subject if more channels are bad

% ---- ICA ----------------------------------------------------------------
cfg.icaType             = 'runica';      % 'runica' (extended Infomax) | 'picard'
cfg.icaEpochWindow      = [-200 1000];   % ms
cfg.icaPreRejAmp        = 500;           % +/- uV; drops only extreme epochs
cfg.icaMinSamplesFactor = 20;            % need k * rank^2 samples for epoched ICA
cfg.iclabelClasses      = {'Muscle','Eye','Heart','Line Noise','Channel Noise'};
cfg.iclabelReviewThresh = 0.7;           % export figures for review
cfg.iclabelRemoveThresh = 0.9;           % auto-remove
cfg.exportICFigures     = true;

% ---- Final epochs / artifact detection ----------------------------------
cfg.erpEpochWindow = [-200 800];   % ms
cfg.baselineWindow = [-200 0];     % ms
cfg.ampThresh      = 100;          % +/- uV, absolute threshold
cfg.p2pThresh      = 100;          % uV, moving-window peak-to-peak
cfg.p2pWinSize     = 200;          % ms
cfg.p2pWinStep     = 50;           % ms

% ---- Execution ----------------------------------------------------------
cfg.nWorkers = 4;    % 1 = run serially
%% ========================================================================

%% SETUP
eeglab nogui;
set(0, 'DefaultFigureVisible', 'off');

cfg.setDir = fullfile(cfg.outputDir, 'sets');
cfg.erpDir = fullfile(cfg.outputDir, 'erp');
cfg.qcDir  = fullfile(cfg.outputDir, 'qc');
validateConfig(cfg);
cellfun(@ensureDir, {cfg.outputDir, cfg.setDir, cfg.erpDir, cfg.qcDir});

if cfg.recursive
    files = dir(fullfile(cfg.dataDir, '**', cfg.filePattern));
else
    files = dir(fullfile(cfg.dataDir, cfg.filePattern));
end
nFiles = numel(files);
if nFiles == 0
    error('No files matching "%s" found in %s.', cfg.filePattern, cfg.dataDir);
end
fprintf('Found %d file(s).\n', nFiles);
saveProvenance(cfg);

%% RUN
qcAll = cell(nFiles, 1);
useParallel = cfg.nWorkers > 1 && nFiles > 1 && license('test', 'Distrib_Computing_Toolbox');

if useParallel
    pool = gcp('nocreate');
    if isempty(pool), pool = parpool(min(cfg.nWorkers, nFiles)); end
    wait(parfevalOnAll(pool, @() set(0, 'DefaultFigureVisible', 'off'), 0));
    wait(parfevalOnAll(pool, @() maxNumCompThreads(1), 0));

    futures(1:nFiles) = parallel.FevalFuture;
    for i = 1:nFiles
        futures(i) = parfeval(pool, @processSubject, 1, files(i), cfg);
    end
    for k = 1:nFiles
        try
            [idx, qc] = fetchNext(futures);
            qcAll{idx} = qc;
            fprintf('%s', futures(idx).Diary);
        catch ME
            fprintf(2, 'Worker failure: %s\n', ME.message);
        end
        fprintf('--- %d/%d finished ---\n', k, nFiles);
    end
else
    for i = 1:nFiles
        qcAll{i} = processSubject(files(i), cfg);
        fprintf('--- %d/%d finished ---\n', i, nFiles);
    end
end

writeBatchSummary(qcAll, cfg.qcDir);
fprintf('\nDone. Outputs in: %s\n', cfg.outputDir);


%% ======================= SUBJECT-LEVEL PIPELINE =========================
function qc = processSubject(fileEntry, cfg)
    filePath = fullfile(fileEntry.folder, fileEntry.name);
    subj     = getSubjectId(fileEntry, cfg);
    qc       = initQC(subj, filePath);
    fprintf('\n=== %s ===\n', subj);

    if ~cfg.overwrite && isfile(fullfile(cfg.setDir, [subj '_preprocessed.set']))
        qc.status = 'skipped';
        fprintf('Output exists, skipping (cfg.overwrite = false).\n');
        return;
    end

    try
        %% 1-2. Load, drop non-EEG channels, check locations
        EEG = eeg_checkset(loadRaw(filePath));
        labels   = {EEG.chanlocs.labels};
        isNonEEG = ismember(upper(labels), upper(cfg.nonEEGChans));
        if any(isNonEEG)
            EEG = pop_select(EEG, 'nochannel', find(isNonEEG));
        end
        if ~isempty(cfg.chanlocLookup)
            EEG = pop_chanedit(EEG, 'lookup', cfg.chanlocLookup);
        end
        assertChanlocs(EEG);
        nChan = EEG.nbchan;
        qc.nChannels = nChan;

        %% 3. Sanity checks
        nEvents = numel(EEG.event);
        if ~isempty(cfg.minEventCount) && nEvents < cfg.minEventCount
            qc = excludeSubject(qc, cfg, sprintf( ...
                'Only %d events (minimum %d): possible trigger failure.', nEvents, cfg.minEventCount));
            return;
        end

        chanStd = std(double(EEG.data(:, :)), 0, 2);
        flatIdx = find(chanStd < cfg.flatStdThresh)';
        if numel(flatIdx) / nChan >= cfg.maxPropFlat
            qc = excludeSubject(qc, cfg, sprintf( ...
                '%.0f%% of channels flat: global recording failure.', 100 * numel(flatIdx) / nChan));
            return;
        end

        %% 4. Bad-channel detection on a 1 Hz high-passed, resampled copy
        EEG_det = pop_basicfilter(EEG, 1:nChan, 'Boundary', 'boundary', ...
            'Cutoff', cfg.hpICA, 'Design', 'butter', 'Filter', 'highpass', ...
            'Order', cfg.filterOrder, 'RemoveDC', 'on');
        EEG_det = pop_resample(EEG_det, cfg.fsNew);
        [badIdx, detFlags] = detectBadChannels(EEG_det, flatIdx, cfg);
        qc.flags       = [qc.flags, detFlags];
        qc.badChannels = {EEG.chanlocs(badIdx).labels};
        clear EEG_det

        if numel(badIdx) / nChan > cfg.maxPropBadChans
            qc = excludeSubject(qc, cfg, sprintf( ...
                '%d/%d channels bad (limit %.0f%%).', numel(badIdx), nChan, 100 * cfg.maxPropBadChans));
            return;
        end

        %% 5-6. Build both branches, assign bins
        EEG_ica = assignBins(buildBranch(EEG, badIdx, cfg.hpICA, cfg), cfg);
        EEG_erp = assignBins(buildBranch(EEG, badIdx, cfg.hpERP, cfg), cfg);
        clear EEG

        %% 7. Rank estimation
        theoreticalRank = EEG_ica.nbchan - numel(badIdx) - 1;   % interpolation + average ref
        empiricalRank   = estimateRank(EEG_ica.data);
        icaRank         = min(theoreticalRank, empiricalRank);
        qc.rank = struct('theoretical', theoreticalRank, 'empirical', empiricalRank, 'used', icaRank);
        if theoreticalRank ~= empiricalRank
            qc.flags{end+1} = sprintf(['Theoretical rank (%d) differs from empirical rank (%d); ' ...
                'used %d. Check reference handling.'], theoreticalRank, empiricalRank, icaRank);
        end

        %% 8. ICA (epoched, with continuous fallback)
        EEG_icaEp = pop_epochbin(EEG_ica, cfg.icaEpochWindow, 'none');
        EEG_icaEp = pop_artextval(EEG_icaEp, 'Channel', 1:EEG_icaEp.nbchan, 'Flag', 1, ...
            'Threshold', [-1 1] * cfg.icaPreRejAmp, 'Twindow', epochLimitsMs(EEG_icaEp));
        good       = ~EEG_icaEp.reject.rejmanual;
        minSamples = cfg.icaMinSamplesFactor * icaRank^2;
        nSamples   = sum(good) * EEG_icaEp.pnts;

        if nSamples >= minSamples
            if any(~good)
                EEG_icaEp = pop_rejepoch(EEG_icaEp, find(~good), 0);
            end
            icaIn = EEG_icaEp;
            qc.icaMode = 'epoched';
        else
            icaIn = EEG_ica;
            qc.icaMode = 'continuous';
            qc.flags{end+1} = sprintf(['Continuous-data ICA fallback: %d clean epoched samples ' ...
                '< %d required.'], nSamples, minSamples);
        end
        clear EEG_icaEp EEG_ica

        icaIn = pop_runica(icaIn, icaArguments(cfg.icaType, icaRank){:});

        %% 9. ICLabel
        icaIn     = pop_iclabel(icaIn, 'default');
        icClasses = icaIn.etc.ic_classification.ICLabel.classes;
        probs     = icaIn.etc.ic_classification.ICLabel.classifications;
        classIdx  = find(ismember(icClasses, cfg.iclabelClasses));
        reviewIdx = find(any(probs(:, classIdx) >= cfg.iclabelReviewThresh, 2))';
        removeIdx = find(any(probs(:, classIdx) >= cfg.iclabelRemoveThresh, 2))';
        qc.icReview  = reviewIdx;
        qc.icRemoved = removeIdx;
        qc.icProbabilitiesReviewed = probs(reviewIdx, :);

        if cfg.exportICFigures && ~isempty(reviewIdx)
            try
                exportICReview(icaIn, reviewIdx, probs, icClasses, fullfile(cfg.qcDir, subj));
            catch figME
                qc.flags{end+1} = ['IC figure export failed (' figME.message ...
                    '): inspect ICs manually before trusting auto-removal.'];
            end
        end

        %% 10. Transfer ICA solution, clean, low-pass, epoch
        EEG_erp.icaweights  = icaIn.icaweights;
        EEG_erp.icasphere   = icaIn.icasphere;
        EEG_erp.icawinv     = icaIn.icawinv;
        EEG_erp.icachansind = icaIn.icachansind;
        EEG_erp.etc.ic_classification = icaIn.etc.ic_classification;
        EEG_erp = eeg_checkset(EEG_erp, 'ica');
        clear icaIn

        if ~isempty(removeIdx)
            EEG_erp = pop_subcomp(EEG_erp, removeIdx, 0);
        end

        EEG_erp = pop_basicfilter(EEG_erp, 1:EEG_erp.nbchan, 'Boundary', 'boundary', ...
            'Cutoff', cfg.lpERP, 'Design', 'butter', 'Filter', 'lowpass', 'Order', cfg.filterOrder);
        EEG_erp = pop_epochbin(EEG_erp, cfg.erpEpochWindow, 'none');
        EEG_erp = pop_rmbase(EEG_erp, cfg.baselineWindow);

        %% 11. Artifact detection
        twin = epochLimitsMs(EEG_erp);
        EEG_erp = pop_artextval(EEG_erp, 'Channel', 1:EEG_erp.nbchan, 'Flag', [1 2], ...
            'Threshold', [-1 1] * cfg.ampThresh, 'Twindow', twin);
        EEG_erp = pop_artmwppth(EEG_erp, 'Channel', 1:EEG_erp.nbchan, 'Flag', [1 3], ...
            'Threshold', cfg.p2pThresh, 'Twindow', twin, ...
            'Windowsize', cfg.p2pWinSize, 'Windowstep', cfg.p2pWinStep);

        %% 12. Average, QC, save
        ERP = pop_averager(EEG_erp, 'Criterion', 'good', 'ExcludeBoundary', 'on', 'SEM', 'on');
        qc.trialsAcceptedPerBin = ERP.ntrials.accepted;
        qc.trialsRejectedPerBin = ERP.ntrials.rejected;
        lowBins = find(ERP.ntrials.accepted < cfg.minTrialsPerBin);
        if ~isempty(lowBins)
            qc.flags{end+1} = sprintf('Bin(s) below %d accepted trials: %s', ...
                cfg.minTrialsPerBin, mat2str(lowBins));
        end

        pop_saveset(EEG_erp, 'filename', [subj '_preprocessed.set'], 'filepath', cfg.setDir);
        pop_savemyerp(ERP, 'erpname', [subj '_avg'], 'filename', [subj '_avg.erp'], ...
            'filepath', cfg.erpDir, 'Warning', 'off');

        qc.status = 'ok';
        writeQC(cfg.qcDir, qc);
        fprintf('Finished %s | ICA: %s | rank %d | bad chans %d | ICs removed %d\n', ...
            subj, qc.icaMode, icaRank, numel(badIdx), numel(removeIdx));

    catch ME
        qc.status = 'error';
        qc.exclusionReason = sprintf('%s (%s, line %d)', ME.message, ME.stack(1).name, ME.stack(1).line);
        writeQC(cfg.qcDir, qc);
        fprintf(2, 'ERROR %s: %s\n', subj, qc.exclusionReason);
    end
end


%% ======================= PROCESSING HELPERS =============================
function EEG = loadRaw(filePath)
    [folder, name, ext] = fileparts(filePath);
    switch lower(ext)
        case '.set',                 EEG = pop_loadset('filename', [name ext], 'filepath', folder);
        case '.cdt',                 EEG = pop_loadcurry(filePath);
        case '.vhdr',                EEG = pop_loadbv(folder, [name ext]);
        case {'.bdf', '.edf', '.gdf'}, EEG = pop_biosig(filePath);
        otherwise
            error('Unsupported file format "%s". Add a loader in loadRaw().', ext);
    end
end

function [badIdx, flags] = detectBadChannels(EEG, flatIdx, cfg)
    % All three detectors run on the SAME channel set, so indices stay aligned.
    flags  = {};
    labels = {EEG.chanlocs.labels};

    [~, kurtIdx] = pop_rejchan(EEG, 'elec', 1:EEG.nbchan, 'threshold', cfg.kurtosisThresh, ...
        'norm', 'on', 'measure', 'kurt');

    corrIdx = [];
    try
        EEG_c = pop_clean_rawdata(EEG, 'FlatlineCriterion', 'off', ...
            'ChannelCriterion', cfg.chanCorrThresh, 'LineNoiseCriterion', 'off', ...
            'Highpass', 'off', 'BurstCriterion', 'off', 'WindowCriterion', 'off');
        corrIdx = find(~ismember(labels, {EEG_c.chanlocs.labels}));
    catch ME
        flags{end+1} = ['Correlation-based detection skipped (' ME.message ...
            '); used flat-line + kurtosis only.'];
    end

    badIdx = unique([flatIdx(:); kurtIdx(:); corrIdx(:)])';
end

function EEG = buildBranch(EEG, badIdx, hpCutoff, cfg)
    EEG = pop_basicfilter(EEG, 1:EEG.nbchan, 'Boundary', 'boundary', ...
        'Cutoff', hpCutoff, 'Design', 'butter', 'Filter', 'highpass', ...
        'Order', cfg.filterOrder, 'RemoveDC', 'on');
    EEG = pop_resample(EEG, cfg.fsNew);
    if ~isempty(badIdx)
        EEG = pop_interp(EEG, badIdx, 'spherical');
    end
    EEG = pop_reref(EEG, []);
end

function EEG = assignBins(EEG, cfg)
    EEG = pop_creabasiceventlist(EEG, 'AlphanumericCleaning', 'on', ...
        'BoundaryNumeric', {-99}, 'BoundaryString', {'boundary'}, 'Warning', 'off');
    EEG = pop_binlister(EEG, 'BDF', cfg.binDescriptorFile, 'IndexEL', 1, ...
        'SendEL2', 'EEG', 'Voutput', 'EEG', 'Warning', 'off');
end

function r = estimateRank(data)
    % Rank from eigenvalues of the channel covariance matrix.
    X  = double(data(:, :));
    X  = X - mean(X, 2);
    ev = eig((X * X') / size(X, 2));
    r  = sum(ev > max(ev) * 1e-7);
end

function args = icaArguments(icaType, rank)
    switch lower(icaType)
        case 'runica', args = {'icatype', 'runica', 'extended', 1, 'pca', rank, 'interrupt', 'off'};
        case 'picard', args = {'icatype', 'picard', 'pca', rank};
        otherwise,     error('Unknown cfg.icaType "%s".', icaType);
    end
end

function exportICReview(EEG, icIdx, probs, classes, filePrefix)
    n    = numel(icIdx);
    grid = ceil(sqrt(n));
    locs = EEG.chanlocs(EEG.icachansind);
    act  = reshape(eeg_getica(EEG, icIdx), n, []);

    figTopo = figure('Visible', 'off', 'Position', [50 50 1400 1000]);
    figSpec = figure('Visible', 'off', 'Position', [50 50 1400 1000]);
    for p = 1:n
        ic = icIdx(p);
        [pMax, cMax] = max(probs(ic, :));
        ttl = sprintf('IC%d: %s %.2f', ic, classes{cMax}, pMax);

        set(0, 'CurrentFigure', figTopo); subplot(grid, grid, p);
        topoplot(EEG.icawinv(:, ic), locs, 'electrodes', 'off');
        title(ttl, 'FontSize', 8);

        [spec, freqs] = spectopo(act(p, :), 0, EEG.srate, 'plot', 'off');
        keep = freqs >= 2 & freqs <= 50;
        set(0, 'CurrentFigure', figSpec); subplot(grid, grid, p);
        plot(freqs(keep), spec(keep)); xlim([2 50]);
        title(ttl, 'FontSize', 8);
    end
    saveas(figTopo, [filePrefix '_IC_topos.png']);
    saveas(figSpec, [filePrefix '_IC_spectra.png']);
    close([figTopo figSpec]);
end


%% ======================= UTILITIES ======================================
function validateConfig(cfg)
    assert(isfolder(cfg.dataDir), 'cfg.dataDir not found: %s', cfg.dataDir);
    assert(isfile(cfg.binDescriptorFile), 'Bin descriptor file not found: %s', cfg.binDescriptorFile);
    assert(cfg.hpICA > cfg.hpERP, 'cfg.hpICA should be higher than cfg.hpERP.');
    assert(cfg.lpERP < cfg.fsNew / 2, 'cfg.lpERP must be below Nyquist (cfg.fsNew/2).');
    assert(cfg.baselineWindow(1) >= cfg.erpEpochWindow(1) && cfg.baselineWindow(2) <= cfg.erpEpochWindow(2), ...
        'cfg.baselineWindow must lie within cfg.erpEpochWindow.');
    assert(cfg.iclabelRemoveThresh >= cfg.iclabelReviewThresh, ...
        'cfg.iclabelRemoveThresh should be >= cfg.iclabelReviewThresh.');
end

function id = getSubjectId(fileEntry, cfg)
    [~, id] = fileparts(fileEntry.name);
    if ~isempty(cfg.subjectIdRegex)
        tok = regexp(fullfile(fileEntry.folder, fileEntry.name), cfg.subjectIdRegex, 'match', 'once');
        if ~isempty(tok), id = tok; end
    end
end

function assertChanlocs(EEG)
    hasLoc = arrayfun(@(c) ~isempty(c.X), EEG.chanlocs);
    if ~all(hasLoc)
        error('Channels without locations: %s. Set cfg.chanlocLookup.', ...
            strjoin({EEG.chanlocs(~hasLoc).labels}, ', '));
    end
end

function t = epochLimitsMs(EEG)
    t = [EEG.xmin EEG.xmax] * 1000;
end

function qc = initQC(subj, filePath)
    qc = struct('subject', subj, 'file', filePath, 'status', 'running', ...
        'exclusionReason', '', 'nChannels', NaN, 'badChannels', {{}}, ...
        'rank', struct(), 'icaMode', '', 'icReview', [], 'icRemoved', [], ...
        'icProbabilitiesReviewed', [], 'trialsAcceptedPerBin', [], ...
        'trialsRejectedPerBin', [], 'flags', {{}});
end

function qc = excludeSubject(qc, cfg, reason)
    qc.status = 'excluded';
    qc.exclusionReason = reason;
    writeQC(cfg.qcDir, qc);
    fprintf('EXCLUDED %s: %s\n', qc.subject, reason);
end

function writeQC(qcDir, qc)
    fid = fopen(fullfile(qcDir, [qc.subject '_qc.json']), 'w');
    fprintf(fid, '%s', jsonencode(qc, 'PrettyPrint', true));
    fclose(fid);
end

function writeBatchSummary(qcAll, qcDir)
    qcAll = qcAll(~cellfun(@isempty, qcAll));
    n = numel(qcAll);
    [subject, status, reason] = deal(cell(n, 1));
    [nBadChans, rankUsed, nICsRemoved, minTrials, nFlags] = deal(nan(n, 1));
    for i = 1:n
        q = qcAll{i};
        subject{i}     = q.subject;
        status{i}      = q.status;
        reason{i}      = q.exclusionReason;
        nBadChans(i)   = numel(q.badChannels);
        nICsRemoved(i) = numel(q.icRemoved);
        nFlags(i)      = numel(q.flags);
        if isfield(q.rank, 'used'),         rankUsed(i)  = q.rank.used; end
        if ~isempty(q.trialsAcceptedPerBin), minTrials(i) = min(q.trialsAcceptedPerBin); end
    end
    T = table(subject, status, nBadChans, rankUsed, nICsRemoved, minTrials, nFlags, reason);
    writetable(T, fullfile(qcDir, 'batch_qc_summary.csv'));
    fprintf('\nSummary: %d ok, %d excluded, %d errors, %d skipped.\n', ...
        sum(strcmp(status, 'ok')), sum(strcmp(status, 'excluded')), ...
        sum(strcmp(status, 'error')), sum(strcmp(status, 'skipped')));
end

function saveProvenance(cfg)
    prov = struct('config', cfg, 'matlab', version, 'date', char(datetime('now')));
    try, prov.eeglab = eeg_getversion; catch, prov.eeglab = 'unknown'; end
    fid = fopen(fullfile(cfg.qcDir, 'config_used.json'), 'w');
    fprintf(fid, '%s', jsonencode(prov, 'PrettyPrint', true));
    fclose(fid);
end

function ensureDir(d)
    if ~isfolder(d), mkdir(d); end
end
