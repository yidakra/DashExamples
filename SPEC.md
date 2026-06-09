# CastleRAG Technical Specification

## Scope

CastleRAG is a multimodal RAG system for verifiable multiple-choice question answering over the CASTLE 2024 dataset, targeting the CASTLE Challenge @ EgoVis 2026 benchmark. The initial goal is a working offline pipeline that:

1. reads the official CASTLE dataset layout from Hugging Face,
2. preprocesses main and auxiliary modalities into aligned retrieval units,
3. embeds those units with `Tevatron/OmniEmbed-v0.1-multivent`,
4. indexes them in Qdrant with filterable payload,
5. retrieves and reranks evidence for a natural-language question,
6. generates one of the four answer choices with citations,
7. exports predictions and computes accuracy when a ground-truth answer file is available.

This spec deliberately targets a reliable first end-to-end system, not the most ambitious challenge solution. The top systems in the 2026 challenge used heavier agentic and summarization stacks; this design keeps the same overall retrieval-first shape but stays closer to the existing `rainrag` architecture so implementation risk is manageable.

## Source Constraints

- CASTLE 2024 main data is organized as `main/day{1..4}/{camera}/{video,transcript,metadata}`. Each hour is a separate file such as `08.mp4`, `08.json`, and `08.*.csv`. Missing hours may appear as `.novideo`. The dataset paper states that videos are one-hour, time-aligned segments and that recording gaps inside otherwise present hours are padded with a test-card placeholder.
- CASTLE 2024 auxiliary data is organized separately under `auxiliary/{gaze,heartrate,photo,thermal,video}`.
- Official transcripts are JSON files with `chunks`, each containing `[start, end]` timestamps and text. Timestamps are relative to the enclosing hour.
- OmniEmbed-multivent is a shared embedding model across text, audio, image, and video, built on Qwen2.5-Omni-7B. The model card shows text queries formatted as `Query: ...` and raw media passed through the Qwen Omni processor.
- Qdrant supports JSON payload, payload filtering, and payload indexes. Fields used for filtering should be indexed explicitly.
- The CASTLE Codabench page describes the question file as a JSON object keyed by question id, with a `query` string and four answer options under `answers`. Submission output is a JSON mapping from question id to `a|b|c|d`. Accuracy is exact-match over all questions.

## 1. Repository Structure

Proposed repository layout:

- `README.md`: replace the placeholder Dash text with project setup, dataset expectations, and pipeline commands.
- `SPEC.md`: this spec.
- `pyproject.toml`: package metadata and dependencies.
- `configs/base.yaml`: default local configuration.
- `configs/snellius.yaml`: Snellius-specific paths, SLURM defaults, and Qdrant settings.
- `scripts/slurm/`: batch templates for preprocessing, embedding, indexing, reranking, and evaluation.
- `data/manifests/`: generated manifests for discovered CASTLE assets and derived chunks.
- `data/derived/`: chunk JSONL or Parquet outputs, keyframes, clip manifests, and optional visual summaries.
- `src/castlerag/config.py`: Pydantic config models and config loader.
- `src/castlerag/cli.py`: Typer CLI entrypoint.
- `src/castlerag/dataset/layout.py`: CASTLE path discovery, naming rules, and camera metadata.
- `src/castlerag/dataset/transcripts.py`: transcript JSON parsing and absolute timestamp alignment.
- `src/castlerag/dataset/metadata.py`: hourly sensor CSV loaders from `main/.../metadata`.
- `src/castlerag/preprocess/windows.py`: sliding-window creation for main video chunks.
- `src/castlerag/preprocess/media.py`: ffmpeg-based subclip and keyframe extraction.
- `src/castlerag/preprocess/auxiliary.py`: photo, auxiliary video, thermal, heartrate, and gaze normalization.
- `src/castlerag/preprocess/visual_summary.py`: offline visual summaries for chunks using the selected open-weight VL model.
- `src/castlerag/schemas.py`: shared typed models for chunk records, retrieval hits, rerank results, and eval items.
- `src/castlerag/embed/omniembed.py`: OmniEmbed processor and batch inference wrappers.
- `src/castlerag/index/qdrant.py`: collection creation, payload indexes, deterministic ids, and batched upserts.
- `src/castlerag/retrieval/search.py`: query encoding, modality-scoped Qdrant search, and score fusion.
- `src/castlerag/retrieval/filters.py`: day, camera, participant, room, time range, and modality filters.
- `src/castlerag/rerank/llm_reranker.py`: local LLM-as-reranker prompts and scoring.
- `src/castlerag/generation/answer.py`: answer generation prompt, citation formatting, and answer extraction.
- `src/castlerag/eval/io.py`: loaders for official questions, local answer keys, and submission export.
- `src/castlerag/eval/run_eval.py`: full benchmark loop and accuracy computation.
- `tests/`: unit and integration tests.

## 2. Data Preprocessing Pipeline

### 2.1 Raw Inputs

Inputs come from two branches:

- Main branch:
  - `main/day{1..4}/{camera}/video/{HH}.mp4`
  - `main/day{1..4}/{camera}/transcript/{HH}.json`
  - `main/day{1..4}/{camera}/metadata/{HH}.*.csv`
- Auxiliary branch:
  - `auxiliary/heartrate/{participant}/...`
  - `auxiliary/gaze/*.csv`
  - `auxiliary/photo/{participant}/*`
  - `auxiliary/thermal/*`
  - `auxiliary/video/{participant}/*`

### 2.2 Canonical Time Model

All derived records must use:

- `day`: `day1` to `day4`
- `hour`: integer 8 to 20 from the source filename
- `start_seconds` and `end_seconds`: offsets within the hour
- `absolute_start` and `absolute_end`: `day + hour + offset`
- `camera_id`: exact folder name, e.g. `Allie`, `Kitchen`, `Living1`
- `camera_type`: `ego` for participant cameras, `fixed` for room cameras
- `participant_id`: participant name for ego cameras, null for fixed cameras

This avoids the `rainrag` assumption that each source file is already a single document-level retrieval unit.

### 2.3 Main Video Windowing

Primary retrieval windows:

- window size: 30 seconds
- stride: 15 seconds
- overlap: 15 seconds

Reasoning:

- 30 seconds is short enough for tractable video embedding and reranking.
- 15-second stride reduces boundary misses for events that cross window edges.
- At 600 hours total, this yields about `600 * 3600 / 15 = 144,000` main windows before filtering.

Filtering rules:

- skip `.novideo` hours completely
- mark windows as `is_placeholder=true` when more than 80% of sampled frames match the CASTLE test-card placeholder
- do not embed placeholder windows into the main retrieval index
- keep windows with no transcript if real video exists; these remain visual-only evidence

### 2.4 Transcript Alignment

For each hourly transcript JSON:

1. parse `chunks[*].timestamp` and `chunks[*].text`
2. convert timestamps from hour-relative to absolute
3. assign transcript chunks to each 30-second window by temporal overlap
4. concatenate overlapping text in time order
5. preserve the original segment list in payload for auditability

Derived transcript fields per window:

- `transcript_text`
- `transcript_segments`
- `has_speech`
- `transcript_char_len`

### 2.5 Keyframe Sampling

For each retained 30-second main window:

- extract 8 JPEG keyframes at uniform offsets
- default offsets: `0s, 4s, 8s, 12s, 16s, 20s, 24s, 28s`
- store them under `data/derived/keyframes/{day}/{camera}/{hour}/{chunk_id}/`

These keyframes are used for:

- debugging and manual evidence inspection
- optional visual summary generation
- future frontend playback previews

### 2.6 Subclip Extraction

For each retained main window:

- extract a 30-second MP4 subclip with audio
- keep original frame rate unless ffmpeg decode becomes a bottleneck; if needed, use a derived retrieval copy at 2 fps for embedding while retaining the original reference path

Stored paths:

- `source_video_path`
- `retrieval_clip_path`
- `keyframe_paths`

### 2.7 Visual Summary Generation

Reranking will consume text, not raw video. To support visual questions, preprocessing includes an offline textual summary step using the same open-weight VL model selected for reranking and generation.

For each main window:

- input: 8 keyframes plus the transcript text if present
- output: 2-4 sentence `scene_summary`
- include visible people, objects, room cues, and obvious actions
- do not infer beyond what is visible or spoken

This summary is stored alongside the chunk and becomes part of the reranker and generator evidence pack.

### 2.8 Auxiliary Modality Handling

Auxiliary data is normalized into standalone retrievable records plus optional links back to nearby main windows.

#### Heartrate

- create 60-second summary records per participant
- fields: `bpm_mean`, `bpm_min`, `bpm_max`, `bpm_delta_prev`
- create a text rendering such as `Heartrate for Allie at day2 14:03-14:04: mean 92 bpm, rising from 86 bpm`
- embed as text modality

#### Gaze

- parse each participant CSV
- create 10-second summary records only for intervals with gaze rows
- keep simple first-pass features: row count, mean x/y, std x/y, valid sample ratio
- create a text rendering such as `Gaze session for Bjorn at day1 10:15:00-10:15:10: stable fixation around center-left`
- embed as text modality

This is intentionally simple because the exact gaze columns must be confirmed against the raw files.

#### Photos

- one record per image
- extract timestamp from EXIF when present, otherwise fall back to filename time pattern
- store original path and optional OCR text
- embed as image modality

#### Thermal

- one record per BMP image
- use file order plus any available metadata for timestamping
- embed as image modality

#### Auxiliary Video

- one record per video file if duration <= 30 seconds
- otherwise re-window into 30-second clips with 15-second stride
- embed as video modality

### 2.9 Output Format Per Chunk

Main windows are written as JSONL or Parquet records with at least:

- `chunk_id`
- `parent_source_id`
- `source_type` (`main_video`, `aux_photo`, `aux_video`, `aux_thermal`, `aux_heartrate`, `aux_gaze`)
- `modality`
- `day`
- `hour`
- `camera_id`
- `camera_type`
- `participant_id`
- `room`
- `start_seconds`
- `end_seconds`
- `absolute_start`
- `absolute_end`
- `source_video_path`
- `retrieval_clip_path`
- `keyframe_paths`
- `transcript_path`
- `transcript_text`
- `transcript_segments`
- `scene_summary`
- `has_speech`
- `is_placeholder`
- `linked_aux_ids`
- `version`

## 3. Indexing Pipeline

### 3.1 Points Written to Qdrant

CastleRAG writes multiple retrievable points into one shared collection:

- one `text_transcript` point per main window
- one `video_audio` point per main window
- one point per auxiliary asset or auxiliary summary record

This keeps the retrieval space shared while allowing modality filters. It also avoids a hard choice between transcript-only and video-only indexing.

Approximate point counts for the first build:

- main windows: about 144,000
- transcript points: about 144,000
- video points: about 144,000
- auxiliary points: expected low tens of thousands
- total: roughly 300,000 to 350,000 points

### 3.2 OmniEmbed Batching Strategy

Batch separately by modality:

- text batches: 64 records
- image batches: 16 records
- video batches: 4 records
- audio batches: only if introduced later as standalone points

Implementation rules:

- discover embedding dimensionality from the first successful batch and use it when creating the Qdrant collection
- keep one SLURM array shard per modality and day to simplify retries
- cache intermediate embeddings to disk before Qdrant upsert
- make point ids deterministic: `sha1(model_version + source_type + chunk_id + modality)`

### 3.3 Qdrant Collection and Payload Schema

Collection name:

- `castle_multimodal_v1`

Payload fields stored with every point:

- `point_id`
- `chunk_id`
- `parent_source_id`
- `source_type`
- `modality`
- `day`
- `hour`
- `camera_id`
- `camera_type`
- `participant_id`
- `room`
- `start_seconds`
- `end_seconds`
- `absolute_start`
- `absolute_end`
- `duration_seconds`
- `transcript_text`
- `scene_summary`
- `asset_path`
- `keyframe_paths`
- `transcript_path`
- `has_speech`
- `is_placeholder`
- `linked_aux_ids`
- `model_name`
- `model_revision`
- `build_id`

Create Qdrant payload indexes for:

- `day`
- `camera_id`
- `camera_type`
- `participant_id`
- `room`
- `modality`
- `source_type`
- `absolute_start`
- `absolute_end`
- `has_speech`

### 3.4 SLURM Job Structure

Jobs:

- `preprocess_main.slurm`: discover files, build 30-second windows, extract subclips and keyframes
- `preprocess_aux.slurm`: normalize heartrate, gaze, photo, thermal, and auxiliary video
- `visual_summary.slurm`: generate `scene_summary` text for main windows
- `embed_text.slurm`: transcript and auxiliary text records
- `embed_video.slurm`: main-window video clips and auxiliary video clips
- `embed_images.slurm`: photos and thermal images
- `index_qdrant.slurm`: collection creation, payload index creation, and batched upsert

Recommended Snellius partition:

- `gpu_a100`

Relevant official Snellius facts:

- `gpu_a100` exposes one GPU as a quarter-node job with 18 CPU cores, 120 GiB RAM, and a billing weight of 128 SBU.

### 3.5 Runtime and SBU Estimate

This section is an estimate, not a measured benchmark. The OmniEmbed model card does not publish A100 throughput, so the numbers below are planning assumptions to validate cluster budget.

Assumptions:

- 144,000 main windows
- one transcript embedding plus one video embedding per main window
- video embedding dominates runtime
- video embedding throughput lands between 2.5 and 3.5 seconds per 30-second clip on one A100 after warmup

Estimated runtime:

- text and auxiliary embeddings: 2 to 4 GPU-hours total
- video embeddings: 100 to 140 GPU-hours total
- full embedding phase: 105 to 145 GPU-hours total
- on 8 concurrent A100 GPUs: about 14 to 19 wall-clock hours

Estimated Snellius cost:

- 1 A100 GPU-hour = 128 SBU
- 105 to 145 GPU-hours = 13,440 to 18,560 SBU
- official rate is EUR 15 per 1,000 SBU
- total embedding cost estimate = about EUR 202 to EUR 278

Add 10 to 20% headroom for retries, cold starts, and visual summary generation.

## 4. Retrieval Pipeline

### 4.1 Query Input

The system accepts:

- question text
- optional explicit filters:
  - `day`
  - `camera_id`
  - `participant_id`
  - `room`
  - `modality`
  - `time_range`

This is an explicit API contract. The first working pipeline should not rely on the LLM to infer those filters from free text.

### 4.2 Query Encoding

Encode the user query as text with OmniEmbed using the model-card query convention:

- `Query: {question}`

For multiple-choice questions, encode both:

- the bare question
- `Question + options` as a second embedding text

Use the max score across the two query forms during fusion. This helps when the answer choices contain concrete objects or names missing from the question stem.

### 4.3 Qdrant Search Strategy

Run separate filtered searches and fuse them:

- transcripts: top 40
- main video clips: top 40
- photos: top 10
- auxiliary videos: top 10
- heartrate: top 5
- gaze: top 5
- thermal: top 5

Fusion:

- Reciprocal Rank Fusion with `k=60`

Why separate searches instead of one global search:

- transcript points would otherwise dominate because they are denser and cheaper to retrieve
- auxiliary modalities need guaranteed exposure to the reranker
- per-modality budgets make evidence coverage more predictable

### 4.4 Filtering

Qdrant filters are applied server-side using payload indexes for:

- exact day
- exact camera
- exact participant
- exact modality
- room
- absolute time overlap

Time overlap rule:

- retrieve points where `absolute_end >= query_start` and `absolute_start <= query_end`

### 4.5 Retrieval Output Format

Each hit returned to the reranker contains:

- `rank`
- `score`
- `point_id`
- `chunk_id`
- `source_type`
- `modality`
- `day`
- `camera_id`
- `participant_id`
- `absolute_start`
- `absolute_end`
- `transcript_text`
- `scene_summary`
- `asset_path`

## 5. Reranking

### 5.1 Model Choice

Use one open-weight model for both reranking and generation.

Default:

- `Qwen2.5-VL-7B-Instruct`

Fallback:

- `InternVL2-8B`

The model is run locally on Snellius or an equivalent GPU host.

### 5.2 Candidate Representation

Each retrieved item is converted to text before reranking. Format:

```text
Candidate {rank}
Source type: {source_type}
Modality: {modality}
Day: {day}
Camera: {camera_id}
Participant: {participant_id or N/A}
Time: {absolute_start} to {absolute_end}
Transcript: {transcript_text or "[no speech]"}
Visual summary: {scene_summary or "[not available]"}
```

This is the minimum viable representation that still lets the reranker reason about visual-only clips.

### 5.3 Reranker Prompt Template

```text
You are ranking evidence for a multiple-choice question over a multimodal lifelog dataset.

Question:
{question}

Answer choices:
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Candidate evidence:
{candidate_text}

Score this candidate on two axes:
1. Evidence relevance from 0 to 4
2. Support for each answer choice from 0 to 4

Return strict JSON:
{
  "relevance": 0-4,
  "support": {"a": 0-4, "b": 0-4, "c": 0-4, "d": 0-4},
  "keep": true|false,
  "rationale": "<<=40 words>"
}
```

### 5.4 Scoring Mechanism

For each candidate:

- parse JSON
- compute `final_rerank_score = 0.7 * relevance + 0.3 * max_support`
- discard candidates with `keep=false` or `relevance <= 1`
- retain top 12 candidates globally

Also compute answer priors:

- sum `support.a`, `support.b`, `support.c`, `support.d` across kept candidates

These priors are passed to generation as soft evidence, not as the final answer.

### 5.5 Feed Into Generation

Generation receives:

- top 12 reranked evidence items
- per-choice cumulative support scores
- the original retrieval scores

## 6. Generation

### 6.1 Prompt Template

```text
You answer multiple-choice questions about the CASTLE dataset.

Rules:
- Use only the provided evidence.
- Prefer direct evidence over speculation.
- If evidence is weak, say so briefly but still choose the most supported option.
- Every factual claim used in the decision must cite at least one evidence item.
- Citations must use the format [camera={camera_id} time={day} {start}-{end}] or [aux={source_type} id={chunk_id}].
- End with exactly one line: FINAL_ANSWER: a|b|c|d

Question:
{question}

Choices:
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Choice support priors:
{support_summary}

Evidence:
{top_reranked_evidence}
```

### 6.2 Citation Format

Main video evidence:

- `[camera=Allie time=day2 14:03:00-14:03:30]`

Auxiliary evidence:

- `[aux=photo id=photo_day2_allie_00034]`
- `[aux=heartrate id=hr_day2_allie_1403]`

### 6.3 Handling the 4-Choice Format

The model sees the four options verbatim and must choose one of `a`, `b`, `c`, or `d`.

Post-processing:

- parse the last line with regex `FINAL_ANSWER:\s*([abcd])`
- if parsing fails, fall back to the highest support prior from reranking
- log the raw answer text for auditability

### 6.4 Evaluation Output

For challenge submission:

- write `submissions.json`
- format:

```json
{
  "2026_q1": "a",
  "2026_q2": "c"
}
```

## 7. Evaluation

### 7.1 Metric

- accuracy = `correct / total_questions`

No partial credit.

### 7.2 Official Question Loader

Support the official CASTLE question JSON shape:

```json
{
  "2026_q1": {
    "query": "Question text",
    "answers": {
      "a": "First answer",
      "b": "Second answer",
      "c": "Third answer",
      "d": "Fourth answer"
    }
  }
}
```

### 7.3 Local Evaluation Inputs

The eval runner should accept:

- `questions_path`: official question JSON
- `answers_path`: local answer key if available
- `predictions_path`: optional cached predictions

If `answers_path` is missing:

- still run the full prediction pass
- export `submissions.json`
- do not claim an accuracy number

### 7.4 Full Eval Pass

Per question:

1. load question and options
2. retrieve candidates
3. rerank
4. generate final choice
5. save prediction plus evidence trace

Outputs:

- `outputs/predictions.json`
- `outputs/evidence_traces.jsonl`
- `outputs/submissions.json`
- `outputs/metrics.json` when ground truth exists

## 8. What Is Reusable From `rainrag`

### 8.1 Reusable With Minimal Changes

- `src/rainrag/config.py`
  - Reuse the hierarchical Pydantic config pattern.
  - Replace VTT-specific and provider-specific fields with CASTLE dataset, OmniEmbed, SLURM, and local VL model settings.

- `src/rainrag/cli.py`
  - Reuse the Typer command layout and lazy imports.
  - Replace commands with `preprocess`, `embed`, `index`, `retrieve`, `answer`, and `eval`.

- `src/rainrag/index.py`
  - Reuse Qdrant connection handling, deterministic point ids, collection creation, and batched upsert.
  - Extend payload schema and add payload-index creation.

- `tests/unit/test_config.py`, `tests/unit/test_index.py`, `tests/unit/test_cli.py`
  - Reuse testing style and expected failure-path coverage.

- `src/eval/run_eval.py`
  - Reuse the idea of a separate evaluation CLI.
  - Replace synthetic dataset generation with official question loading.

### 8.2 Reusable Conceptually But Not Directly

- `src/rainrag/query.py`
  - Reuse the orchestration shape: encode query, retrieve, rerank, prompt, answer.
  - Replace text-only assumptions, online API providers, and Cohere rerank with local multimodal retrieval plus local LLM reranking.

- `src/rainrag/api.py`
  - Reuse request/response model ideas later if an API is exposed.
  - Do not port before the offline pipeline works.

### 8.3 Not Reusable / Direct Conflicts

- `src/rainrag/ingest.py`
  - Conflict: assumes each file becomes one text document or VTT chunk set.
  - CASTLE requires multimodal alignment across hour videos, transcripts, metadata CSVs, and auxiliary assets.

- `src/rainrag/embed.py`
  - Conflict: built around sentence-transformers and text embeddings.
  - CastleRAG needs OmniEmbed multimodal inference and modality-specific batching.

- `src/rainrag` hybrid BM25 path
  - Conflict: rainrag uses BM25 over transcript text as a first-class retrieval channel.
  - CastleRAG’s first working pipeline should stay centered on OmniEmbed + Qdrant + modality-aware fusion. Transcript BM25 can be added later if recall is poor.

- web metadata, MCP server, Streamlit UI, and journalistic answer shaping
  - Irrelevant for the initial CastleRAG objective.

## 9. Open Questions and Risks

### 9.1 Dataset and Ground Truth

- It is unclear whether the local workspace will include an answer key for the 185 questions. Without it, the pipeline can export predictions but cannot compute accuracy offline.
- Auxiliary timestamp quality needs validation, especially for thermal and personal media.

### 9.2 Throughput and Storage

- OmniEmbed-multivent A100 throughput is not documented in the model card. Runtime and SBU estimates in this spec are planning numbers and must be replaced with measured benchmark logs.
- 8 keyframes per 144,000 windows produces roughly 1.15 million JPEGs. Storage and inode pressure need to be managed carefully.

### 9.3 Placeholder and Gap Detection

- The dataset includes placeholder padding inside hour videos. If detection is weak, the index will contain many useless windows and retrieval quality will degrade.

### 9.4 Gaze and Metadata Semantics

- The exact meaning of several hourly metadata files and gaze CSV columns is not established by the currently inspected source material. The first implementation should not overfit to guessed semantics.

### 9.5 Reranker Evidence Bottleneck

- The top CASTLE systems explicitly treated evidence selection as the bottleneck.
- WDL’s abstract emphasizes question-type routing, ASR chunk retrieval, attaching auxiliary images, and candidate frame sampling.
- MARS’s abstract emphasizes source selection across transcripts, video, gaze, heartrate, photos, and thermal, plus long-video compression through captioning and summaries.
- This means a transcript-only port of `rainrag` is very likely to fail on visual, temporal, or auxiliary-modality questions even if the generation model is strong.

### 9.6 Scope Risk

- A full challenge-style agentic planner is out of scope for the first implementation.
- The intended first milestone is a dependable retrieval-plus-rerank baseline with citations, not a leaderboard-optimized multi-agent system.
