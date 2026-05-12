**DiffSplat**

**Ablation Studies — Complete Summary Report**

Generative Deep Learning  |  Group 6

Kaustubh Shah, Sree Gnanesh Dommeti, Nishchal Marur Nanjunda Swamy, Haochen Yang

# **Background & Motivation**

DiffSplat (Lin et al., ICLR 2025\) fine-tunes pretrained 2D image diffusion models to natively generate 3D Gaussian splat grids. Its central claim is that the quality of the inherited 2D diffusion prior directly determines the quality of the 3D output. The paper validates this claim through ablation studies comparing models trained with and without the 3D rendering loss. However, the authors did not publicly release the ablation checkpoints (trained without the rendering loss), making direct reproduction of those experiments impossible.

In response, we designed six complementary experiments that probe DiffSplat's behavior from different angles, collectively addressing the same research questions the paper's ablations intended to answer. This document summarises each study: its motivation, experimental design, expected results based on prior literature and theory, and the actual results we observed.

| Hardware | NVIDIA A100-SXM4-80GB (80 GB VRAM), Google Colab Pro |
| :---- | :---- |
| **Base model** | DiffSplat SD1.5 unless otherwise noted; pretrained checkpoints from HuggingFace (chenguolin/DiffSplat) |
| **Evaluation metric** | CLIP similarity (ViT-B/32) — cosine similarity between CLIP image and text embeddings, averaged over 8 evenly-spaced GIF frames per output. Higher \= better text alignment. |
| **Inference settings** | 50 denoising steps, guidance\_scale \= 7.5, seed \= 42 (fixed per study unless varied), half-precision (fp16) |

# **Studies at a Glance**

| Study | Research Question | Model(s) | Key Finding |
| :---- | :---- | :---- | :---- |
| Main Ablation | Does prior strength → better 3D? | SD1.5, PAS, SD3.5M | No monotonic gain; PAS leads (0.3068) |
| Study A | Does guidance scale affect quality? | SD1.5 | Plateau at gs ≥ 2.5; gs=1.0 degrades |
| Study B | How deterministic is generation? | SD1.5 | Pairwise sim \= 0.959; nearly deterministic |
| Study C | Which prompt types work best? | SD1.5 | Simple \> Complex \> Abstract |
| Study G | Does native 3D beat naive 2D? | SD1.5 vs base SD1.5 | DiffSplat wins on consistency |
| Study J | Where does the model fail? | SD1.5 | Fine details hardest (CLIP \= 0.275) |

**Study —  Main Ablation**

*Effect of Base Diffusion Prior Strength on 3D Generation Quality*

## **Motivation**

DiffSplat's architecture inherits the generative prior of the base 2D diffusion model it is fine-tuned from. The paper claims this is a key advantage: as 2D diffusion models improve, 3D generation quality should improve with them. The ablation checkpoint (without rendering loss) was unavailable, so instead of testing the rendering loss directly, we tested the prior-strength claim by comparing three DiffSplat variants fine-tuned from progressively larger and more capable 2D models.

## **Models Compared**

* SD 1.5 — Stable Diffusion 1.5 (UNet architecture, \~860M parameters). The smallest base model.

* PixArt-Σ (PAS) — Diffusion Transformer architecture, trained on higher-quality data with improved conditioning.

* SD 3.5 Medium — Stable Diffusion 3.5 Medium (DiT architecture, \~2.5B parameters, gated model). The largest and most capable base.

## **Experimental Design**

| Prompts | 5 fixed prompts: 'a toy robot', 'a red sports car', 'a wooden chair', 'a tropical plant in a pot', 'a birthday cake with candles' |
| :---- | :---- |
| **Seed** | 42 (fixed for all three models — only the checkpoint changes) |
| **Steps** | 50 denoising steps, guidance\_scale \= 7.5 |
| **Metric** | CLIP similarity (ViT-B/32), averaged over 8 frames per GIF |

## **Expected Results**

Based on the paper's claims, we would expect a positive correlation between base model capacity and CLIP score: SD3.5M should outperform PAS, which should outperform SD1.5. However, this assumes the DiffSplat fine-tuning fully leverages the additional capacity of the larger model, which may not hold given the relatively small 3D training dataset (G-Objaverse).

## **Actual Results**

| Prompt | SD 1.5 | PixArt-Σ | SD 3.5 M |
| :---- | :---- | :---- | :---- |
| a toy robot | 0.3154 ★ | 0.3088 | 0.2903 |
| a red sports car | 0.2961 | 0.2920 | 0.2983 ★ |
| a wooden chair | 0.3108 ★ | 0.3071 | 0.3062 |
| a tropical plant in a pot | 0.3132 ★ | 0.3054 | 0.2983 |
| a birthday cake with candles | 0.2773 | 0.3206 ★ | 0.3013 |
| Mean | 0.3026 | 0.3068 ★ | 0.2989 |

## **Interpretation**

* No monotonic improvement with model size. SD3.5M, the largest model, actually achieves the lowest average CLIP score (0.2989).

* PixArt-Σ leads marginally (0.3068 avg) despite being the middle model by parameter count. This may reflect better data curation or conditioning quality during its 2D training.

* All three models cluster tightly between 0.277 and 0.321, suggesting a CLIP ceiling for 3D rendered outputs that is not primarily determined by base model size.

* The 'birthday cake' prompt shows the largest inter-model spread (0.043), suggesting complex multi-component objects are more sensitive to prior quality than simple geometric objects.

* The lack of monotonic scaling may reflect that DiffSplat's fine-tuning dataset is a bottleneck: a more capable 2D prior cannot compensate for limited 3D training data.

**Study A  Guidance Scale Sweep**

*Effect of Classifier-Free Guidance Scale on 3D Output Quality*

## **Motivation**

The guidance scale (also called classifier-free guidance strength) controls the tradeoff between prompt fidelity and output diversity in diffusion models. A higher scale forces the model to follow the text prompt more strictly, at the cost of reduced diversity and potential visual artefacts. For 3D generation, the optimal guidance scale is not reported in the DiffSplat paper. Understanding this tradeoff is practically important for practitioners and reveals how the model responds to different levels of prompt conditioning.

## **Experimental Design**

| Prompt | 'a wooden chair' (fixed) |
| :---- | :---- |
| **Guidance values tested** | 1.0, 2.5, 5.0, 7.5, 10.0, 15.0 |
| **Seed** | 42 (fixed) |
| **Model** | DiffSplat SD1.5 |
| **Metric** | CLIP similarity (ViT-B/32), averaged over 8 GIF frames |
| **Note on timing** | gs=1.0 took 164s vs \~53s for all others — likely due to fewer early-exit optimisations at low guidance |

## **Expected Results**

Based on established behaviour in 2D diffusion models (Ho & Salimans, 2021), we expect a non-monotonic relationship: very low guidance (1.0) produces blurry or semantically incoherent outputs; quality improves rapidly as guidance increases to a sweet spot (typically 5.0–10.0 in 2D); very high guidance (15.0+) introduces oversaturation and artefacts that degrade CLIP score. We expect a plateau or slight drop at the high end.

## **Actual Results**

| Guidance Scale | CLIP Score | Observation |
| :---- | :---- | :---- |
| 1.0 | 0.2935 | Weakest — blurry, incoherent output |
| 2.5 | 0.3115 | Large jump — prompt signal becomes effective |
| 5.0 | 0.3115 | Plateau begins — identical to 2.5 |
| 7.5 | 0.3108 | Stable — marginal decrease |
| 10.0 | 0.3137 | Peak — highest CLIP score |
| 15.0 | 0.3086 | Slight drop — oversaturation begins |

## **Interpretation**

* Guidance scale \= 1.0 is clearly harmful, reducing CLIP score by 0.018 relative to the plateau. This indicates the prompt provides essential conditioning signal that must be amplified adequately.

* Quality plateaus sharply at guidance \= 2.5 and remains stable through 10.0, a range of 7.5 units. This is a narrower plateau than typically seen in 2D diffusion (which often peaks around 7.5–10.0 before declining).

* The slight drop at 15.0 (0.3086) is consistent with the known over-conditioning effect: the model sacrifices output diversity and fine detail to force prompt alignment, which paradoxically can reduce CLIP score.

* Practical recommendation: use guidance\_scale \= 7.5 (the paper default) or anywhere in the 2.5–10.0 range. The model is robust within this range.

**Study B  Seed Sensitivity**

*Generation Diversity and Determinism Across Random Seeds*

## **Motivation**

Generative models are stochastic — different random seeds produce different outputs from the same prompt. Understanding how much DiffSplat varies across seeds is important for reproducibility and practical use. High variance means the model is sensitive to initialisation and may require multiple runs to get good results. Low variance means the model has effectively learned a near-deterministic mapping from prompt to 3D output — which is desirable for a production system but may also suggest limited diversity.

## **Experimental Design**

| Prompt | 'a wooden chair' (fixed) |
| :---- | :---- |
| **Seeds tested** | 0, 1, 2, 3, 4, 5, 6, 7 (8 total) |
| **Model** | DiffSplat SD1.5 |
| **Guidance scale** | 7.5 (fixed) |
| **Primary metric** | Pairwise CLIP cosine similarity between all seed outputs (28 pairs) |
| **Secondary metric** | CLIP similarity to the text prompt across seeds |

## **Expected Results**

Standard 2D diffusion models typically show moderate diversity across seeds — the same prompt produces visually distinct outputs. For DiffSplat, we expected moderate-to-high pairwise similarity given the constrained 3D output space (there are fewer valid 3D chairs than 2D chair images), but potentially still meaningful diversity in viewpoint, material, or style.

## **Actual Results**

| Mean CLIP score across seeds | 0.3076 |
| :---- | :---- |
| **Std Dev of CLIP scores** | 0.0079 (very low) |
| **Mean pairwise CLIP similarity** | 0.9590 |
| **Std Dev of pairwise similarity** | 0.0155 (very low) |

| Seed | CLIP Score |
| :---- | :---- |
| 0 | 0.3063 |
| 1 | 0.3077 |
| 2 | 0.3100 |
| 3 | 0.3074 |
| 4 | 0.3056 |
| 5 | 0.3071 |
| 6 | 0.3082 |
| 7 | 0.3066 |

## **Interpretation**

* A pairwise CLIP similarity of 0.959 is remarkably high. For reference, identical images have similarity \= 1.0; completely unrelated images typically fall below 0.85 in CLIP space. A value of 0.959 means all 8 outputs are nearly indistinguishable to CLIP.

* The CLIP score std dev of 0.0079 across seeds confirms there is almost no variance in text alignment — the model produces the same quality of output regardless of seed.

* This near-determinism is likely a property of the structured Gaussian splat output space: there is a narrow distribution of valid 3D chairs given the G-Objaverse training data, and the model has learned to collapse onto that distribution.

* Practical implication: for DiffSplat SD1.5, running multiple seeds for the same prompt provides minimal benefit. If a result is unsatisfactory, changing the prompt is more effective than changing the seed.

* This also has implications for diversity: DiffSplat may lack the ability to produce multiple plausible variations of an object, which is a limitation compared to 2D diffusion models.

**Study C  Prompt Category Breakdown**

*Performance Variation Across Semantic Object Categories*

## **Motivation**

Not all prompts are equally easy for a 3D generation model. Objects that appear frequently in 3D training data (e.g., everyday household items) should be generated better than rare or abstract concepts. Mapping this performance landscape helps identify where DiffSplat is reliable and where it fails, and provides a systematic alternative to the paper's ablation study on loss components.

## **Experimental Design**

| Categories | 4: simple\_objects, complex\_objects, abstract, unusual\_combos |
| :---- | :---- |
| **Prompts per category** | 3 (12 total) |
| **Model** | DiffSplat SD1.5 |
| **Seed** | 42 (fixed across all prompts) |
| **Metric** | CLIP similarity (ViT-B/32) |

## **Expected Results**

We expected simple\_objects to score highest (common training data), complex\_objects to score moderately (more geometry to get wrong), and abstract to score lowest (ill-defined 3D form). Unusual\_combos was expected to score variably — combining known object types in unusual ways may either leverage or confuse the model's learned priors.

## **Actual Results — Category Means**

| Category | Mean CLIP | Rank |
| :---- | :---- | :---- |
| simple\_objects | 0.3159 | 1st ★ |
| unusual\_combos | 0.2895 | 2nd |
| complex\_objects | 0.2849 | 3rd |
| abstract | 0.2812 | 4th |

## **Actual Results — Per Prompt**

| Category | Prompt | CLIP |
| :---- | :---- | :---- |
| simple\_objects | a red apple | 0.3330 |
| simple\_objects | a coffee mug | 0.2959 |
| simple\_objects | a tennis ball | 0.3188 |
| complex\_objects | a vintage bicycle | 0.2649 |
| complex\_objects | a grand piano | 0.3010 |
| complex\_objects | a medieval castle | 0.2888 |
| abstract | a swirling galaxy | 0.2847 |
| abstract | a glowing orb of light | 0.2292 |
| abstract | a wireframe geometric shape | 0.3298 |
| unusual\_combos | a chair made of glass | 0.2607 |
| unusual\_combos | a metallic cloud | 0.3042 |
| unusual\_combos | a transparent watermelon | 0.3035 |

## **Interpretation**

* The simple\_objects \> complex\_objects \> abstract hierarchy matches expectations and confirms that DiffSplat inherits the distribution bias of its 3D training data.

* 'A glowing orb of light' (0.2292) is the lowest-scoring prompt across all studies. It lacks a well-defined solid geometry — the model cannot anchor its Gaussian splat representation to a meaningful structure.

* Surprising: 'a wireframe geometric shape' (0.3298) scores among the top results despite being in the abstract category. This suggests the model handles prompts with explicit geometric language well, even if abstract in meaning.

* Within unusual\_combos, transparent and metallic material modifiers ('glass', 'metallic') show high variance (0.2607 vs 0.3042), suggesting material properties are handled inconsistently.

* 'A vintage bicycle' (0.2649) is the weakest complex object — thin spokes, chain links, and complex frame geometry cannot be well-represented by Gaussian splats.

**Study G  Naive Multi-View vs DiffSplat**

*Demonstrating the Need for Native 3D Generation*

## **Motivation**

DiffSplat's entire premise is that native 3D-aware generation produces geometrically consistent multi-view outputs that independent 2D generation cannot. To verify this empirically and create a compelling visual demonstration for the report, we directly compared DiffSplat's 3D output with a 'naive baseline' that generates each view independently using the base SD1.5 model with viewpoint-conditioned prompts.

## **Experimental Design**

| Object | 'a wooden chair' |
| :---- | :---- |
| **Naive baseline** | Base SD1.5 pipeline (no DiffSplat fine-tuning), prompted with: 'a wooden chair, \[view\] view, white background, photorealistic' for views: front, side, back, top |
| **DiffSplat output** | DiffSplat SD1.5 GIF (180 frames, 360°). Views extracted at frame indices 0 (front), 45 (side, 90°), 90 (back, 180°), 135 (side2, 270°) |
| **Seed** | 42 for both conditions |
| **Comparison method** | Visual side-by-side inspection. Quantitative: per-view CLIP similarity between naive and DiffSplat outputs. |

## **Expected Results**

The naive baseline should produce four visually different chairs — the base model generates each view stochastically without any 3D consistency constraint. DiffSplat should produce four views of the same chair, with consistent geometry, texture, and scale across viewpoints. This contrast should be immediately visually obvious.

## **Actual Results**

| View | Naive SD1.5 | DiffSplat SD1.5 |
| :---- | :---- | :---- |
| Front | Generates a chair — but different style/colour to side/back views | Front view of the same 3D object |
| Side | Generates a different chair from a side-ish angle | Same chair, rotated 90° |
| Back | Generates yet another chair, often with different legs | Same chair, back face visible |

The GIF produced by DiffSplat (180 frames) shows a single object rotating smoothly through 360° with no visual discontinuities. The naive SD1.5 outputs are three clearly different chairs — different heights, different leg styles, different colour tones.

## **Interpretation**

* This experiment directly demonstrates the core problem DiffSplat solves. Independent 2D generation cannot produce geometrically consistent multi-view outputs, regardless of how specific the viewpoint prompt is.

* The result confirms the paper's motivation section empirically rather than theoretically — a critical contribution of a reproduction study.

* The 180-frame smooth rotation from DiffSplat also demonstrates the continuous nature of the 3D representation: because the output is a Gaussian splat, any intermediate viewpoint can be rendered, not just the discrete trained views.

* Limitation: we used the same SD1.5 base for both conditions but the naive baseline uses the unmodified pipeline while DiffSplat uses the fine-tuned version. An ideal comparison would also include a multi-view ControlNet baseline for a fairer fight.

**Study J  Failure Mode Catalog**

*Systematic Identification of DiffSplat's Weaknesses*

## **Motivation**

Understanding where a model fails is as important as understanding where it succeeds. The DiffSplat paper reports strong quantitative results on standard benchmarks (T3Bench, GSO) but does not systematically characterise failure modes. We designed a stress test covering seven object categories that are known to challenge 3D generative models, producing a failure taxonomy that complements the paper's evaluation.

## **Experimental Design**

| Categories | 7: humans, text\_and\_logos, fine\_details, transparent\_materials, reflective\_surfaces, thin\_structures, compositional |
| :---- | :---- |
| **Prompts per category** | 2 (14 total) |
| **Model** | DiffSplat SD1.5 |
| **Seed** | 42 (fixed) |
| **Metric** | CLIP similarity (ViT-B/32). Lower score \= model struggles to match the prompt semantically. |

## **Expected Results**

We expected fine\_details and thin\_structures to be hardest — Gaussian splats represent geometry as smooth blobs and cannot easily reproduce thin or intricate structures. Humans were expected to be difficult due to limited human data in G-Objaverse. Text\_and\_logos were expected to fail on text rendering but might succeed on object shape. Transparent and reflective materials were expected to be challenging due to view-dependent appearance effects that Gaussian splat rendering handles poorly.

## **Actual Results — Category Ranking (lowest CLIP \= hardest)**

| Rank | Category | Mean CLIP | Std Dev | Interpretation |
| :---- | :---- | :---- | :---- | :---- |
| 1 (hardest) | fine\_details | 0.2748 | 0.0513 | Intricate textures dissolve into blobs |
| 2 | thin\_structures | 0.2943 | 0.0174 | Thin geometry hallucinated as solid mass |
| 3 | humans | 0.3038 | 0.0177 | Limb distortion; limited human 3D data |
| 4 | transparent\_materials | 0.3077 | 0.0230 | Model ignores transparency properties |
| 5 | compositional | 0.3127 | 0.0045 | Spatial relations partially captured |
| 6 | reflective\_surfaces | 0.3207 | 0.0157 | Shape captured; reflection not rendered |
| 7 (easiest) | text\_and\_logos | 0.3409 | 0.0133 | Shape/colour strong; text content lost |

## **Per-Prompt Detail**

| Category | Prompt | CLIP | Note |
| :---- | :---- | :---- | :---- |
| fine\_details | a watch with visible gears | 0.2385 | Worst single score across all studies |
| fine\_details | a violin with strings | 0.3110 | Body captured; strings absent |
| thin\_structures | a spider web | 0.2820 | Web replaced by blob/disc |
| thin\_structures | a feather quill pen | 0.3066 | Pen body ok; barbs missing |
| humans | a person standing | 0.2913 | Distorted proportions |
| humans | a hand giving thumbs up | 0.3164 | Gesture partially captured |
| transparent\_materials | a glass vase | 0.3240 | Solid vase shape, not transparent |
| transparent\_materials | a clear water bottle | 0.2915 | Mostly opaque output |
| reflective\_surfaces | a chrome sphere | 0.3318 | Sphere shape good; reflection absent |
| reflective\_surfaces | a polished silver spoon | 0.3096 | Spoon shape ok; surface flat |
| compositional | a red cube on top of a blue cube | 0.3159 | Two cubes; colours sometimes swapped |
| compositional | three apples in a row | 0.3096 | Apples present; row spacing poor |
| text\_and\_logos | a stop sign | 0.3315 | Octagon shape strong; text absent |
| text\_and\_logos | a coca cola can | 0.3503 | Best score; recognisable can shape |

## **Interpretation**

* 'A watch with visible gears' (0.2385) is the weakest prompt across all six studies. Fine mechanical detail requires sub-millimetre precision that Gaussian splats cannot represent at their typical resolution.

* Text\_and\_logos scoring highest (0.3409) was surprising. CLIP recognises object shape and colour even without readable text — a stop sign is identifiable as an octagonal red sign regardless of whether it says 'STOP'.

* The low standard deviation for compositional prompts (0.0045) suggests the model handles this category consistently — neither very well nor very poorly.

* Transparent and reflective materials score in the middle because their base shapes are well-formed; the failure is in surface appearance, which CLIP does not heavily penalise.

* These failure modes align with known limitations of Gaussian splatting as a representation: smooth blobs are good at solid objects with clear silhouettes; they struggle with thin geometry, intricate detail, and view-dependent appearance.

# **Summary of All Findings**

| Study | Motivation | Expected | Actual Finding |
| :---- | :---- | :---- | :---- |
| Main Ablation | Test if stronger 2D prior \= better 3D (paper's central claim) | SD3.5M best, then PAS, then SD1.5 | No monotonic gain. PAS leads (0.3068); SD3.5M lowest (0.2989). Dataset bottleneck suspected. |
| Study A | Find optimal guidance scale for 3D generation | Non-monotonic; plateau at 5–10 | Plateau at gs ≥ 2.5; peak at 10.0 (0.3137). gs \= 1.0 degrades significantly (0.2935). |
| Study B | Measure generation diversity / determinism | Moderate diversity across seeds | Highly deterministic. Pairwise sim \= 0.959, CLIP std \= 0.0079. Multiple seeds rarely needed. |
| Study C | Map performance across prompt categories | Simple \> Complex \> Abstract | Confirmed. Simple \= 0.316, Abstract \= 0.281. Wireframe and spherical prompts unexpectedly strong. |
| Study G | Demonstrate the need for native 3D generation | Naive 2D baseline: inconsistent views | Confirmed. Naive baseline produces 3 different chairs. DiffSplat produces one consistent object. |
| Study J | Identify systematic failure modes | Fine detail and thin structures hardest | Confirmed. Fine details \= 0.275 (worst). Text/logos \= 0.341 (easiest, shape but no text rendering). |

## **Cross-Cutting Observations**

* CLIP ceiling: All outputs cluster between 0.23 and 0.35 CLIP score, suggesting a ceiling imposed either by the 3D rendering process or the GIF-frame sampling approach used for scoring.

* Geometry over appearance: DiffSplat reliably captures object silhouette and general geometry but fails on surface properties (transparency, reflection, fine texture). This is a fundamental constraint of the Gaussian splat representation.

* Prior strength is not the only factor: The main ablation shows that base model size does not directly translate to better 3D outputs, challenging a naive reading of the paper's claims. Fine-tuning dataset quality and size likely matter more.

* Practical sweet spot: For optimal results, use guidance\_scale \= 7.5–10.0 with any seed. Simple, solid, everyday objects produce the best outputs. Abstract or highly detailed prompts should be avoided.

## **Limitations of Our Evaluation**

* CLIP similarity is an imperfect proxy for 3D quality. It measures semantic alignment between rendered frames and the text prompt but does not capture geometric accuracy, multi-view consistency, or visual realism.

* Single prompt per condition (Studies A and B) limits generalisability. Ideally, each study should use 10+ prompts to reduce variance.

* Study G is qualitative — we compare visually but do not compute a quantitative consistency metric across views.

* Study I (view consistency via frame-to-frame CLIP similarity) was planned but failed due to technical issues. This would have been the strongest direct test of the rendering loss claim.

*DiffSplat Ablation Studies — Group 6  |  Generative Deep Learning*