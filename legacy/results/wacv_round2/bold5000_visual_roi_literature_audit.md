# BOLD5000 visual-cortex encoding audit

## Directly relevant papers

| Work | Visual targets | Evaluation | Relation to CHASMBrain |
|---|---|---|---|
| Oota et al., *Visio-Linguistic Brain Encoding*, COLING 2022 ([paper](https://aclanthology.org/2022.coling-1.11/)) | PPA, LOC, EarlyVis, OPA, RSC; four subjects | 10-fold CV over all trials; Pearson across voxels per image, averaged over images and participants | Same Pearson axis and exact public ROI arrays, but different split and participant aggregation. Their Table 5 is not a direct comparison to the CSI1 repeated-image run. |
| Delavari et al., *CNN-extracted features generate synthetic fMRI responses to unseen images*, Vision Research 2025 ([paper](https://doi.org/10.1016/j.visres.2025.108641)) | Five functional visual ROIs; CSI1--CSI3 | Train on non-repeated images; test on 113 repeated images; Pearson across voxels for each measured repetition | Closest published protocol. Same TYPED/GLMdenoise/RR targets, session z-scoring, subject-specific training and spatial Pearson. |
| Tong et al., *A Comparative Analysis of Visual Encoding Models Based on Classification and Segmentation Task-Driven CNNs*, 2020 ([paper](https://doi.org/10.1155/2020/3467916)) | Same five ROIs; CSI1--CSI3 | Pearson for each voxel across a 500-image test set | Uses the opposite correlation axis; numeric Pearson values cannot be placed in the same table as CHASMBrain's spatial Pearson. |
| Huang et al., *DNNBrain: A Unifying Toolbox for Mapping Deep Neural Networks and Brains*, 2020 ([paper](https://doi.org/10.3389/fncom.2020.580632)) | BOLD5000 ventral temporal cortex | Voxelwise Pearson across images with 10-fold CV; also multivariate PLS and RSA | Useful methodological precedent, but its encoding score is not the CHASMBrain metric. |
| Cheng et al., *Curvature as an Organizing Principle of Mid-level Visual Representation*, 2020 ([paper](https://openreview.net/forum?id=b3d1f39772d74a4c2947bb2186cd4a4f99965ec8)) | PPA and LOC | AlexNet + LASSO; 10-fold CV; voxelwise correlation across images | Relevant visual-cortex encoding model, but incompatible correlation axis and only two ROIs. |
| Chang et al., *BOLD5000*, Scientific Data 2019 ([paper](https://doi.org/10.1038/s41597-019-0052-3)) | PPA, LOC, EarlyVis, OPA, RSC | Reliability, RSA/t-SNE and ROI characterization | Dataset/reference source, not a directly competing encoding benchmark. |

## Paper-compatible CSI1 repeated-image experiment

- Targets: public Release 2.0 `TYPED-FITHRF-GLMDENOISE-RR` ROI arrays.
- Normalization: per-session, per-voxel z-score.
- Development pool: 4,803 images shown once, split into 4,323 fit and 480 validation.
- Final test: all 451 measured responses to 113 repeated images. The repeated images are never used for fitting or model selection.
- Metric: Pearson across voxels for every presentation, then average. Each functional ROI is also scored separately.
- Full-output Stage 1: early-stopped at epoch 15; best epoch 5, validation coarse Pearson 0.195.
- Full-output Stage 2: early-stopped at epoch 23; best epoch 13, validation voxel Pearson 0.109.

| Model | PPA | LOC | EarlyVis | OPA | RSC | Concatenated array |
|---|---:|---:|---:|---:|---:|---:|
| CHASMBrain, full 13,197-voxel output $\rightarrow$ ROI intersection (CSI1) | 0.1365 | 0.1824 | 0.0921 | 0.1929 | 0.0959 | 0.2015 |
| VisualBERT (four-subject mean, 10-fold; different protocol) | 0.141 | 0.187 | 0.128 | 0.188 | 0.120 | 0.153 macro mean |

The presentation-facing table omits the local Ridge diagnostic and compares
CHASMBrain with published methods only.  Ridge results remain in the internal
JSON/summary as a sanity check and are not used as a headline baseline.

The presentation-facing CHASMBrain row now comes from the original full-output
two-stage checkpoint, not the auxiliary model retrained directly on 1,685 ROI
entries. Stage 1 predicts 439 coarse clusters and Stage 2 predicts all 13,197
voxels; functional ROIs are applied only at evaluation. The anatomical visual
mask contains 1,162/1,685 public functional-ROI entries (1,159 unique mapped
voxels), verified from the complete beta-response time series with maximum
absolute mapping error $1.43\times10^{-6}$. Per-ROI mapped counts are PPA 271,
LOC 239, EarlyVis 330, OPA 155, and RSC 167.

The five-ROI macro mean for CHASMBrain is 0.1399. The concatenated-array
Pearson is higher because correlations across the combined vector also capture
between-region response structure; it is not the arithmetic mean of the five ROI
scores. Delavari et al. report CSI1 `r=0.20 (SD=0.10)` over their deduplicated
all-ROI mask. CHASMBrain's concatenated mapped entries yield `r=0.2015
(SD=0.1488)`, which is close but not strictly identical because CHASMBrain's
fixed anatomical visual mask covers only the intersection described above.

The VisualBERT row is contextual only until CHASMBrain is run with their full
10-fold, four-subject protocol. It must not be presented as a head-to-head result.
Within the displayed contextual table, measured CSI1 CHASMBrain is second by
five-ROI macro mean (0.140 vs. VisualBERT 0.153) and best on OPA (0.193 vs.
VisualBERT 0.188). The paper marks column-wise best values in bold and
second-best values with underlining.
