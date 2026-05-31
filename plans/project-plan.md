# The Selective Alignment Project: End-to-End Plan

**Paper title (working, not final):** The Selectivity of RLHF: Mechanistic Sources of Cultural Flattening in Post-Training Alignment

**Document version:** v1.1-WIP, 2026-05-27

**Status: WORK IN PROGRESS. NOT FINAL.** This is a living document. The framing, model suite, hypotheses, and numbers below are the current best version and are expected to change as the project progresses, as sources are re-validated, and as pilot results come in. Nothing in this document is locked. Every claim that rests on an external fact carries a citation; claims without a citation are design decisions, not established results, and are open to revision. Sections marked OPEN QUESTION or TO VERIFY are not yet resolved.

**Audience:** project members, future collaborators, and reviewers who want a single document that explains the whole study from scientific question to publishable claim.

---

## 0. Executive Summary

Post-training alignment changes how a language model behaves. After reinforcement learning from human feedback (RLHF) or related methods, a model refuses some requests it used to answer, replaces specific indigenous terms with generic Western ones, and sometimes appears to lose facts it demonstrably had before alignment. The improvement is selective: the same training that makes the model more helpful on average degrades it on specific kinds of culturally specific content. This selectivity is documented behaviorally but its mechanism inside the network is not understood.

This project asks three nested questions. First, when alignment changes behavior on culturally specific content, does it rewrite the model's internal representations of that content, or does it leave the representations intact and install a late-stage policy that redirects the output? Second, is the answer the same for every type of cultural content, or do different categories (regional facts, indigenous vocabulary, sensitive topics) get treated by different mechanisms? Third, to the extent we can tell from available models, does the effect depend on who performed the alignment and how, or is it intrinsic to the post-training objective?

We answer these questions by building a 3,000-item controlled probe set of Indian cultural knowledge (ICCD-3K), running it through paired model checkpoints (base models and their post-trained versions), and using mechanistic interpretability tools to locate, classify, and causally test the representational changes. Indian culture is the testbed, not the topic: it offers rich documented content, an empirically validated regional structure, and the availability of an Indic-focused post-trained model whose comparison to Western-aligned models is informative.

The five phases are: dataset construction (Phase 1), layer-wise probing (Phase 2), sparse dictionary learning (Phase 3), causal validation through activation interventions (Phase 4), and synthesis (Phase 5). The output is one paper at a mechanistic interpretability venue (NeurIPS, ICLR, or ACL), a public dataset release, and a pre-registered analysis plan.

---

## 1. The Scientific Question

The behavioral fact is established. Aligned models behave differently from their base versions on culturally specific inputs, and the change is not uniformly positive. The mechanistic question, which is open, is what produces that change inside the network. Two opposing hypotheses sit in the current literature.

The first is **representation rewriting**. Alignment gradients propagate through the whole network during fine-tuning and modify the mid-layer subspaces where content is encoded. Under this view the model's internal representation of a cultural fact after alignment is genuinely different from before. There is empirical support that fine-tuning changes internal structure: differential analysis between pre-trained and fine-tuned models finds that middle layers undergo the largest changes during task adaptation (arXiv:2505.17073), and reinforcement-learning fine-tuning has been shown to change the intensity and distribution of internal circuit activations (arXiv:2509.21044).

The second is **late-stage policy gating**, the mechanistic form of the Superficial Alignment Hypothesis. This hypothesis, introduced with LIMA (Zhou et al., 2024, arXiv:2305.11206) and reinforced by URIAL (Lin et al., 2023, arXiv:2312.01552), holds that capabilities and knowledge are learned almost entirely in pretraining and that alignment makes superficial adjustments to how that knowledge is expressed. URIAL found that for most token positions an aligned model's top-ranked token is already within the base model's top few candidates, and that base models with a suitable prompt behave very much like aligned models. Later work formalized the notion of "superficial knowledge" and found that a large part of what alignment adds can be captured by a shallow restyling transformation (arXiv:2502.04602). Independent analysis of alignment as distribution shaping reached the same conclusion: aligned model behavior corresponds to a re-weighted subdistribution of base model behavior rather than new content (arXiv:2406.17692). Under this view the cultural knowledge is still in the network after alignment, and what changed is a late mechanism that intercepts and redirects it.

These two hypotheses can produce identical behavioral outputs, so behavior alone cannot distinguish them. They differ in where in the network the change lives and whether mid-layer content is altered, which is exactly what mechanistic interpretability tools can measure. This is the gap the project fills. A recent survey documents that mechanistic interpretability has only begun to be applied to RLHF and that the relationship between alignment and internal representations is an open research direction (arXiv:2602.11180).

The third question, which we approach with more caution, is whether the effect depends on the alignment recipe and the people who produced it. There is a documented mechanism by which annotator composition could cause cultural flattening. The "Alignment Trilemma" (arXiv:2511.19504) formalizes it: when annotators from different cultural backgrounds disagree about whether a response is good, capturing both views makes the reward model noisy; the standard fix is stronger regularization, which pulls the policy toward the majority preference and erases the minority view. This is consistent with broader findings that RLHF amplifies majority viewpoints and collapses diverse preferences (arXiv:2405.16455), and with the general observation that RLHF embeds the values of whoever provides the feedback. Whether this mechanism is the actual cause of the cultural effects we measure is something our design can speak to only suggestively, for reasons explained in Section 3.

Why the answer matters: if alignment installs a late-stage filter, the underlying knowledge is intact and in principle recoverable through activation-level intervention, and the cultural effect is in some sense a surface property. If alignment rewrites mid-layer representations, the knowledge is genuinely altered and cannot be recovered without retraining. The two cases call for different remediation strategies, which is why locating the mechanism is useful beyond the specific case of Indian culture.

---

## 2. The Hypothesis Space

We do not choose between the hypotheses before running the experiment. The pre-registered analysis plan lists four mechanistic outcomes, each with a distinct measurable signature. These are stated here in plain terms; the exact quantitative thresholds that separate them are defined in the Phase 1 specification and locked in the OSF pre-registration before any model is run.

**Outcome A, pure rewriting.** Mid-layer probes (Phase 2) show reduced extractability of the cultural fact in the aligned model relative to base. Crosscoder analysis (Phase 3) classifies most cultural features as shifted or exclusive rather than preserved. Cross-checkpoint patching (Phase 4) restores correct behavior when mid-layer activations are patched from base into aligned, and late-layer patching does little.

**Outcome B, pure gating.** Mid-layer probes show preserved extractability. Crosscoder analysis finds most cultural features preserved, plus a small set of alignment-exclusive features concentrated in late layers. Late-layer patching restores correct behavior; mid-layer patching does not change the output. This is the signature the Superficial Alignment Hypothesis predicts.

**Outcome C, mixed by axis.** Different cultural categories show different signatures. This is the outcome prior work makes most plausible, because the three axes are designed to probe mechanisms that the literature already treats as distinct (factual recall, lexical mapping, and refusal). If regional facts look like shifting, indigenous vocabulary looks like rewriting, and sensitive topics look like late gating, that pattern is itself the contribution.

**Outcome D, no detectable signature.** None of the methods produce a clear pattern. The pre-registration commits us to reporting this honestly. A null result on a well-motivated question with sound methods remains publishable.

Separately from the mechanism axis (A through D), there is the **annotation-and-recipe question**: does the Indic-focused post-trained model (Sarvam-M) show a different representational signature than the Western-aligned models (Llama-3-Instruct, Gemma-2-it) on the same content? We treat this as an exploratory comparison rather than a clean hypothesis test, because the available models confound several variables at once (Section 3). The pre-registration states the comparison we will run and explicitly states what it can and cannot establish.

---

## 3. The Model Suite

We study paired checkpoints. Each pair has a base model and a post-trained version derived from it, which is what lets us attribute a measured representational change to the post-training step rather than to differences in pretraining. The suite has two roles: a clean core that tests the mechanism question, and an exploratory third pair that addresses the recipe-and-annotation question with appropriate caveats.

**Pair 1, the clean core: Llama-3-8B and Llama-3-8B-Instruct.** Architecturally 32 layers, hidden dimension 4096, vocabulary 128256. The aligned version uses supervised fine-tuning followed by human-preference RLHF. This pair is the primary subject because the refusal-direction literature, including Arditi et al. (2024, arXiv:2406.11717), is benchmarked on these checkpoints, which makes our Axis C results directly comparable to published work.

**Pair 2, architectural robustness: Gemma-2-9B and Gemma-2-9B-it.** A different shape (42 layers, hidden dimension 3584, vocabulary 256000) aligned by a different Western team (Google) with supervised fine-tuning and preference-based reinforcement learning. Including this pair lets us check that any finding from Pair 1 is not an artifact of one architecture or one lab's alignment process.

Pairs 1 and 2 together constitute the defensible core of the study: both are Western-team, human-preference RLHF applied to a primarily-English base. They answer the central question, namely what Western-style preference RLHF does to Indian cultural representations, and whether the answer is the same across two architectures and two labs.

**Pair 3, exploratory contrast: Mistral-Small-3.1-24B-Base-2503 and Sarvam-M.** Mistral-Small-3.1-24B has 40 layers and hidden dimension 5120 (verified against the published Hugging Face configuration). Sarvam-M is a post-trained derivative built by Sarvam AI, an Indian team, on the Mistral base. It is the closest available thing to a "non-Western alignment" of a known Western base, which is why it is in the suite.

The important caveat, which corrects an earlier and too-clean version of this plan: **Sarvam-M is not produced by human-preference RLHF.** According to Sarvam's technical report (sarvam.ai/blogs/sarvam-m), Sarvam-M is built with supervised fine-tuning on a curated set of roughly 5.2 million prompts, followed by reinforcement learning with verifiable rewards (RLVR) using the GRPO algorithm. RLVR rewards verifiable correctness on tasks like mathematics, code, and instruction-following; it does not optimize a learned model of human preference about tone or cultural appropriateness in the way classic RLHF does. Furthermore, the supervised fine-tuning completions were generated using other approved models (which may themselves be Western-aligned) and then filtered, and the team reported explicitly adjusting outputs to reduce bias and improve cultural relevance.

The consequence for inference is that Pair 3 differs from Pairs 1 and 2 along at least three confounded dimensions simultaneously: the team and its cultural context (Indian vs Western), the training-data language mix (Indic-heavy vs English-heavy), and the alignment algorithm (SFT plus RLVR vs human-preference RLHF). Therefore a difference between Sarvam-M and the Western-aligned models cannot be attributed to annotator origin alone. We state this plainly in the paper. A clean isolation of annotator origin would require holding the base model and the alignment algorithm fixed while varying only the annotator pool, and no public model pair provides that. The Sarvam-M comparison is suggestive evidence about whether Indic-focused post-training as a bundle produces a different representational signature, not a controlled manipulation of who annotated.

**TO VERIFY before Phase 4 on Pair 3.** Sarvam reported exploring a vocabulary extension with Indian-language tokens but found it degraded the knowledge base and that the distillation route with vocabulary changes did not beat plain supervised fine-tuning, which strongly suggests the shipped Sarvam-M retained Mistral's original tokenizer. Cross-checkpoint activation patching requires that base Mistral and Sarvam-M tokenize identical text into identical token sequences at identical positions. Before any patching on Pair 3, the pipeline must load both tokenizers, assert equal vocabulary size, and assert identical tokenization on a battery of ICCD-3K prefixes. If the assertion fails, Pair 3 is restricted to representation-level analyses that do not require token-position correspondence (probing and Crosscoder feature classification), and cross-checkpoint patching for Pair 3 is dropped with the reason documented.

All models are open-weight and downloadable from Hugging Face. All are loaded in bfloat16. None require special access agreements. We pin a Hugging Face commit hash for every checkpoint.

---

## 4. Phase 1: Dataset Construction (ICCD-3K)

This is the foundation; everything downstream depends on a clean probe set. The full specification lives in a separate companion document (currently at v1.1-WIP) covering every variable, threshold, source, and filter. This section is the summary needed to follow the rest of the plan.

ICCD-3K is 3,000 minimal pairs. Each pair has a clean prefix containing a specific Indian cultural anchor, a corrupted prefix in which the anchor is replaced, and a target answer (a state name or a topic name). We measure the log-probability the model assigns to the target on each prefix; the difference is the per-item signal. The minimal-pair design follows CounterFact (Meng et al., 2022, arXiv:2202.05262), the standard structure for this kind of mechanistic study.

The set is stratified across three axes, five regions (the INDICA framework, Madhusudan et al., 2026, arXiv:2601.15550), and four sub-concepts per axis, giving 60 cells of 50 items. Fifty items per cell is set by a power analysis for a paired test at medium effect size (Cohen's d = 0.5) with 80% power, not chosen for roundness. The three axes are designed to instantiate the three mechanistic predictions in Section 2: Axis A (Regional Specificity) probes factual recall and is expected to show shifting; Axis B (Cultural Flattening) probes indigenous-to-generic lexical mapping and is expected to show rewriting; Axis C (Sensitive Policy) probes refusal and is expected to show late gating, connecting to the refusal-direction line of work.

Items are sourced through a hybrid pipeline: SANSKRITI (Maji et al., 2025, arXiv:2506.15355) for anthropologically grounded cultural facts, Wikipedia for clean factual sentences and a deterministically built gazetteer used to detect place-name leakage, and a whitelisted web-search fallback only for cells that would otherwise fall below the floor. The counterfactual strategy is under active revision (see the companion spec): the current direction is a cross-anchor swap, where the corrupted prefix names a same-category cultural item from a different region, so that the per-item signal measures specific binding (this festival to this state) rather than the diffuse prior over states. A model-side validation step on Llama-3-8B-base drops items where the base model shows no meaningful difference between the two prefixes; this doubles as a filter and is one of the most important quality gates in the pipeline.

Phase 1 ends with a released, pre-registered, version-locked dataset ready for inference.

---

## 5. Phase 2: Layer-Wise Probing

Before the heavier machinery, we locate where in each model the cultural information lives and whether it is accessible to downstream computation. The unit of analysis is the residual-stream activation at the final prefix token, extracted per item, per layer, per prefix version, per model. We compute four diagnostics.

**Linear probing measures presence.** For each layer we fit a logistic regression to predict the target from the activation. High accuracy means the layer linearly encodes the fact; low accuracy means the fact is absent or non-linearly encoded. This is the standard probing approach from representation analysis.

**Minimum Description Length probing measures accessibility.** Information can be present but practically unusable by the model's own downstream layers. MDL, via online coding (Voita and Titov, 2020, arXiv:2003.12298), measures the code length needed to communicate the labels from the activations; shorter means more readily usable. Comparing MDL between base and aligned is the cleanest test of whether alignment changed how accessible the cultural fact is, as opposed to merely whether it is present.

**Layer-wise KL divergence locates divergence.** Running both checkpoints in a pair and comparing their output distributions at each layer through the Logit Lens shows the layer at which the two models most diverge, pointing to where the alignment effect is concentrated.

**Direct Logit Attribution locates the decision.** Projecting each layer's attention and MLP outputs onto the target-minus-alternative unembedding direction gives a per-component contribution to the answer. This catches a late-stage filter that mid-layer probes would miss.

Read together, the four diagnostics produce a per-axis, per-layer picture: preserved linear accuracy with a late DLA drop is a gating signature; a mid-layer MDL increase is a rewriting signature. Phase 2 output is descriptive and motivates the causal work in Phase 4.

---

## 6. Phase 3: Dictionary Learning

Probing says where information lives, not which features encode it. To reach feature level we decompose the residual stream into a sparse dictionary, using two complementary tools.

**BatchTopK Crosscoders for base-versus-aligned diffing.** A Crosscoder (Lindsey et al., 2024, Anthropic) is a sparse autoencoder trained jointly on two models' residual streams, learning a shared feature dictionary in which each feature carries one decoder vector per model. The relative decoder norm (Δ-norm) classifies each feature as preserved, shifted, or exclusive. We use the BatchTopK sparsity formulation (Bussmann et al., 2024, arXiv:2412.06410), which the chat-tuning Crosscoder work of Minder et al. (2025, arXiv:2504.02922) found gives cleaner attribution than an L1 penalty. We train one Crosscoder per model pair on roughly 500 million tokens of background text (FineWeb-Edu plus IndicCorp v2), a corpus kept separate from ICCD-3K because 3,000 items is far too little to learn a large dictionary. After training, the dictionary is frozen and the ICCD-3K items are projected through it to identify and classify the features that fire on cultural content.

**Skip Transcoders for the MLP function.** A Skip Transcoder (Dunefsky, Chlenski, and Nanda, 2024, arXiv:2406.11944) is a sparse autoencoder that predicts an MLP block's output from its input through a sparse bottleneck plus a linear skip path, exposing what the MLP computes rather than only what is written to the stream. We train these on the MLP blocks that Phase 2's DLA flags as important, and compare base versus aligned to see whether the relevant circuit was destroyed or kept and redirected.

Phase 3 output per model pair: a frozen Crosscoder dictionary with each cultural feature classified, and Skip Transcoders on the MLP blocks most relevant to cultural processing.

---

## 7. Phase 4: Causal Validation

Probing and dictionary learning are correlational. Phase 4 tests causation through activation interventions.

**Cross-checkpoint interchange patching** locates the layer responsible. We patch the base model's residual stream at a chosen layer and token position into the aligned model's forward pass and measure how far the aligned output moves toward the base output. Sweeping layers shows whether restoration happens at mid layers (rewriting) or late layers (gating). This step requires token-position correspondence between the two checkpoints, which is guaranteed for Pairs 1 and 2 and must be verified for Pair 3 (Section 3).

**Path and edge patching** narrow localization to specific attention heads and MLP edges (Wang et al., 2023; Goldowsky-Dill et al., 2023), the technique behind the Indirect Object Identification circuit.

**Latent feature steering** tests specific Crosscoder features. Amplifying a shifted feature or suppressing an exclusive feature in the aligned model, and checking whether correct cultural behavior is restored, establishes the causal role of that feature.

For Axis C specifically, this connects to the refusal-direction literature. Arditi et al. (2024) showed refusal is mediated by a single residual-stream direction across many models; we test whether that same direction mediates refusal on culturally sensitive content. We will also engage the counter-evidence honestly: recent work argues refusal is not fully captured by one direction (the Geometry of Refusal work, arXiv:2502.17420, and a 2026 analysis finding more structure than a single direction, arXiv:2602.02132). Our Axis C analysis reports whether a single direction suffices for cultural refusal or whether multiple directions are needed.

Phase 4 output: a per-axis, per-pair causal map of which intervention reliably moves aligned behavior toward base behavior.

---

## 8. Phase 5: Synthesis

Phase 5 combines the descriptive (Phase 2), structural (Phase 3), and causal (Phase 4) evidence into the paper's claim, in three layers. First, a per-axis verdict on rewriting versus gating. Second, a cross-model comparison: whether the mechanism replicates across the two clean Western-RLHF pairs (architectural and lab generality), and separately the exploratory Sarvam-M contrast, reported with the multi-factor caveat from Section 3. Third, a feature-level catalog of the Crosscoder features causally tied to each behavior.

The pre-registered analysis plan, filed before model-side validation in Phase 1, locks the statistical tests, the multiple-comparisons correction, the cell definitions, and the decision rules separating outcomes A through D. Findings are reported as pre-registered or exploratory according to whether they map to a locked test. This addresses the standard concern that mechanistic interpretability results are post hoc.

---

## 9. Cross-Cutting Concerns

**Reproducibility.** Pinned random seeds, pinned Hugging Face commit hashes for all checkpoints, a timestamped and archived Wikipedia snapshot, and code released under MIT at publication. The pipeline is designed to re-run end-to-end on a single workstation with one A100-class GPU in roughly one week of wall-clock time. (This estimate is provisional and will be revised after the pilot.)

**Pre-registration.** The analysis plan is filed on the Open Science Framework before model-side validation in Phase 1, which is the point where the inferential design begins. It is timestamped and citable.

**Compute budget.** Provisional estimate around 250 GPU-hours on A100-class hardware, dominated by Phase 2 probing and Phase 3 Crosscoder training. Phase 1 is CPU- and API-bound. This number is an estimate and will be updated as phases execute.

---

## 10. Risks Across the Whole Project

**Outcome D, no signal.** Mitigation: the pre-registration commits us to publishing a sound null result.

**Crosscoder training instability.** Sparse dictionaries can collapse or fail to learn monosemantic features. Mitigation: BatchTopK is more stable than L1; we train multiple seeds and check that feature classifications agree across seeds.

**Bare-prefix-on-aligned-model artifacts.** Feeding bare prefixes to aligned models is necessary for token-position-aligned patching but is off-distribution for the aligned model. Mitigation: a chat-template robustness check; if conclusions differ between bare and templated conditions, both are reported. (Documented in the Phase 1 spec.)

**Pair 3 multi-factor confound.** Sarvam-M differs from the Western-aligned models in team, data mix, and algorithm (SFT plus RLVR, not human-preference RLHF), so it cannot isolate annotator origin. Mitigation: we frame Pair 3 as exploratory throughout, state the confound in the abstract and limitations, and do not claim a clean annotation-origin result. This is a correction to an earlier version of this plan that over-claimed the cleanliness of this comparison.

**Pair 3 tokenizer mismatch.** If the shipped Sarvam-M did not retain Mistral's tokenizer, cross-checkpoint patching for Pair 3 is impossible. Mitigation: an explicit tokenizer-equality assertion before patching (Section 3); if it fails, Pair 3 is restricted to analyses that do not need token-position correspondence.

**Sarvam-M is not from-scratch Indian.** It is a post-training of a Western base. Mitigation: we never claim otherwise; the comparison is "Indic-focused post-training of a Western base" versus "Western post-training of comparable bases."

**The cultural framing itself.** We are studying Indian cultural content and are not ourselves the cultural authority. Mitigation: every item is grounded in three sources (SANSKRITI's annotator process, INDICA's regional consensus, Wikipedia), an external South Asian studies reviewer audits the Axis C items before release, and the dataset is released publicly for community correction.

**Refusal-direction single-direction assumption.** The single-direction claim is contested. Mitigation: Axis C reports whether one direction suffices for cultural refusal or whether multiple are needed, citing the contesting work.

---

## 11. What Success Looks Like

Success is three deliverables.

First, a mechanistic answer, with causal evidence, to whether Western-style preference RLHF rewrites or gates Indian cultural representations, resolved per axis. Any of outcomes A through D is publishable.

Second, a feature-level catalog of the Crosscoder features that encode preserved, shifted, and alignment-exclusive cultural content, with named features that future interpretability work can reuse.

Third, an exploratory but carefully bounded comparison of whether Indic-focused post-training (Sarvam-M) leaves a different representational signature than Western preference-RLHF on the same content, with the multi-factor confound stated explicitly so the comparison is read as suggestive rather than dispositive.

The contribution sits in mechanistic interpretability, with the cultural setting as the testbed. Primary venues: NeurIPS, ICLR, ACL.

---

## 12. Known Limitations

We commit to reporting these in the paper.

**English-only.** We test cultural knowledge in English prompts, not in Hindi, Tamil, or other Indian languages. This is shared with SANSKRITI and INDICA and is a constraint of working through model tokenizers that are primarily English.

**Five-region granularity.** South India alone contains Telugu, Tamil, Malayalam, and Kannada cultures with different practices; the five-region analysis cannot see intra-regional variation. We use five regions because INDICA validates them empirically and because they give enough items per region for statistical power.

**Recipe coverage.** Our clean core is human-preference RLHF (Llama, Gemma). We do not separately study DPO, RLAIF, constitutional methods, or RLVR in isolation; Sarvam-M bundles SFT and RLVR and is exploratory. Disentangling alignment recipes is future work.

**Model coverage.** Qwen, DeepSeek, Phi, and others may differ. The current suite is what is tractable on a single-workstation budget with available base-and-aligned pairs.

**Post-training as one event.** We compare base to one post-trained snapshot; we do not study mid-training, continual pretraining, or chained alignment stages.

**Cultural Flattening dichotomy.** Axis B assumes a clean indigenous-versus-generic contrast, which is clean for some concepts (Raga versus scale) and fuzzy for others. We restrict Axis B to concepts with academic documentation of the contrast and accept the coverage cost.

**Selection-on-the-dependent-variable in model-side validation.** Phase 1's validation step keeps items where the base model already shows an effect, which can inflate measured effects on the surviving sample. Mitigation: we report the base-model effect-size distribution so readers can judge whether survivors are atypical. (Detailed in the Phase 1 spec.)

---

## 13. Working Title and Framing Note

The current working title is "The Selectivity of RLHF: Mechanistic Sources of Cultural Flattening in Post-Training Alignment." It leads with the alignment mechanism (the topic), names selectivity (the observation that motivates the work), and names cultural flattening (the phenomenon explained), with Indian culture as the testbed rather than the subject. This positions the work for a mechanistic interpretability audience.

The title is not final and is expected to change once results are in. If the evidence favors one mechanism cleanly, the title may assert it, for example "Cultural Flattening Lives in Late Layers" if gating dominates, or "RLHF Rewrites Cultural Representations in Mid Layers" if rewriting dominates. An earlier working title ("Suppression vs. Shifting") is retired in favor of the RLHF-mechanism-first framing, but the suppression-versus-shifting language may still appear in the body as a description of the two hypotheses.

---

## 14. Document Status and Change Log

**This document is WORK IN PROGRESS and will keep changing as the project advances.** It is the project-level overview. The Phase 1 specification is the detailed companion (currently v1.1-WIP). Phases 2 through 5 will get their own detailed specifications at similar depth, each written and revised before the corresponding phase runs. Nothing here is locked; the OSF pre-registration, filed before model-side validation in Phase 1, is the only artifact that becomes immutable, and only for the inferential design it covers.

**Change log.**

- v1.0 (2026-05-26): initial end-to-end plan under the "Suppression vs. Shifting" framing, with Indian culture as the apparent topic and a clean Western-versus-Indic alignment comparison.
- v1.1-WIP (2026-05-27): reframed the project as an RLHF-mechanism study with Indian culture as testbed. Grounded the two hypotheses in named literature (Superficial Alignment Hypothesis and representation-rewriting evidence). Added the annotation-and-recipe question as an explicitly exploratory axis. Corrected the Pair 3 characterization: Sarvam-M uses SFT plus RLVR (GRPO), not human-preference RLHF, and the comparison is multi-factor and cannot isolate annotator origin; this walks back the over-clean Western-versus-Indic claim in v1.0. Added a required tokenizer-equality verification for Pair 3 before patching. Added honest counter-evidence on the single-direction refusal claim. Marked the entire document WIP. Added a references section.

---

## 15. References (validated for this version)

Citations validated during the v1.1 revision. arXiv identifiers are given where available. This list will grow as the project proceeds; inclusion here means the source was checked against its abstract or text during revision, not that it has been read in full.

- Zhou et al., 2024. LIMA: Less Is More for Alignment. arXiv:2305.11206. (Origin of the Superficial Alignment Hypothesis.)
- Lin et al., 2023. The Unlocking Spell on Base LLMs: Rethinking Alignment via In-Context Learning (URIAL). arXiv:2312.01552.
- (Author list to confirm), 2025. Extracting and Understanding the Superficial Knowledge in Alignment. arXiv:2502.04602. NAACL 2025.
- (Author list to confirm), 2024. From Distributional to Overton Pluralism: Investigating Large Language Model Alignment. arXiv:2406.17692.
- (Author list to confirm), 2025. Mechanistic Interpretability of GPT-like Models on Summarization Tasks. arXiv:2505.17073.
- (Author list to confirm), 2026. Reinforcement Learning Fine-Tuning Enhances Activation Intensity and Diversity in the Internal Circuitry of LLMs. arXiv:2509.21044.
- (Author list to confirm), 2026. Mechanistic Interpretability for Large Language Model Alignment: Progress, Challenges, and Future Directions. arXiv:2602.11180. (Survey.)
- Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda, 2024. Refusal in Language Models Is Mediated by a Single Direction. arXiv:2406.11717. NeurIPS 2024.
- (Author list to confirm), 2025. The Geometry of Refusal in Large Language Models. arXiv:2502.17420. (Contests single-direction.)
- (Author list to confirm), 2026. There Is More to Refusal in Large Language Models than a Single Direction. arXiv:2602.02132. (Contests single-direction.)
- (Author list to confirm), 2025. Position: The Complexity of Perfect AI Alignment, Formalizing the RLHF Trilemma. arXiv:2511.19504.
- (Author list to confirm), 2024. On the Algorithmic Bias of Aligning LLMs with RLHF: Preference Collapse and Matching Regularization. arXiv:2405.16455.
- (Author list to confirm), 2023. Mitigating the Alignment Tax of RLHF. arXiv:2309.06256.
- Sarvam AI, 2025. Sarvam-M: Explorations in Post Training and Inferencing Optimizations for a Hybrid Indic LLM. sarvam.ai/blogs/sarvam-m. (SFT plus RLVR via GRPO; base Mistral-Small-3.1-24B-Base-2503.)
- Meng, Bau, et al., 2022. Locating and Editing Factual Associations in GPT (ROME, CounterFact). arXiv:2202.05262. NeurIPS 2022.
- (Author list to confirm), 2026. CLM-Bench: Benchmarking and Analyzing Cross-lingual Misalignment of LLMs in Knowledge Editing. arXiv:2601.17397. (Closest prior work; Chinese-cultural CounterFact pairs.)
- Bussmann, Leask, Nanda, 2024. BatchTopK Sparse Autoencoders. arXiv:2412.06410.
- Lindsey et al., 2024. Sparse Crosscoders for Cross-Layer Features and Model Diffing. Anthropic (transformer-circuits.pub).
- Minder et al., 2025. (BatchTopK Crosscoders for chat-tuning model diffing.) arXiv:2504.02922.
- Dunefsky, Chlenski, Nanda, 2024. Transcoders Find Interpretable LLM Feature Circuits. arXiv:2406.11944. NeurIPS 2024.
- Voita, Titov, 2020. Information-Theoretic Probing with Minimum Description Length. arXiv:2003.12298. EMNLP 2020.
- Maji et al., 2025. SANSKRITI: A Comprehensive Benchmark for Evaluating Language Models in Indian Culture. arXiv:2506.15355. ACL Findings 2025.
- Madhusudan et al., 2026. Common to Whom? Regional Cultural Commonsense and LLM Bias in India (INDICA). arXiv:2601.15550.

End of plan (v1.1-WIP).