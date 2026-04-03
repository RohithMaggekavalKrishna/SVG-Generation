# SVG-Generation

NYU Tandon — DL Spring 2026 Kaggle Contest: **Text-to-SVG Generation**

Fine-tunes Qwen2.5-7B-Instruct with LoRA on a 135K prompt→SVG dataset, then runs inference on Kaggle without internet access.

---

## Repository Structure

```
SVG-Generation/
├── train_svg.py            # Training script (run on Greene HPC)
├── submit_svg_sft.sh       # SLURM job submission script
├── kaggle_inference.ipynb  # Kaggle submission notebook (inference only)
├── starter-notebook.ipynb  # Reference/tutorial notebook
├── SVG_and_LoRA_Tutorial.ipynb  # SVG + LoRA educational notebook
├── data/
│   └── train.csv           # Competition training data (135 MB)
├── test.csv                # Competition test prompts
├── sample_submission.csv   # Submission format reference
└── requirements.txt        # Python dependencies
```

---

## Training on Greene HPC

### 1. Move train.csv into place

```bash
mkdir -p /home/hk4488/SVG-Generation/data
mv /home/hk4488/SVG-Generation/train.csv /home/hk4488/SVG-Generation/data/train.csv
```

### 2. Submit the job

```bash
cd /home/hk4488/SVG-Generation
sbatch submit_svg_sft.sh
```

Logs: `/scratch/hk4488/SVG-Generation/logs/`
Checkpoints: `/scratch/hk4488/SVG-Generation/outputs/svg_sft/`
LoRA adapter: `/scratch/hk4488/SVG-Generation/outputs/svg_sft/lora/`

### Training config

| Setting | Value |
|---------|-------|
| Base model | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` |
| LoRA rank / alpha | 32 / 32 |
| Batch size (effective) | 16 (2 × 8 accum) |
| Max seq length | 6144 tokens |
| Learning rate | 2e-4 (cosine) |
| Epochs | 1 |
| Hardware | H100 80GB |

---

## Kaggle Submission

### 1. Upload model weights to Kaggle

After training, upload `/scratch/hk4488/SVG-Generation/outputs/svg_sft/lora/` as a Kaggle dataset named **`svg-lora-model`**.

Upload the base model (`Qwen2.5-7B-Instruct`) as a Kaggle dataset named **`svg-base-model`**.

Upload `test.csv` as a Kaggle dataset named **`svg-competition`**.

### 2. Run kaggle_inference.ipynb

Open `kaggle_inference.ipynb` in a Kaggle notebook and run all cells. It will:
1. Install dependencies
2. Load base model + LoRA adapter from `/kaggle/input/`
3. Generate SVGs for all test prompts
4. Validate every output against the competition rules
5. Save `submission.csv`

### 3. Submit

Submit the notebook as a **Code Submission** on Kaggle (3 submissions/day limit).

---

## Scoring

| Component | Weight | Formula |
|-----------|--------|---------|
| Visual Fidelity | 0.85 | 0.7·SSIM + 0.3·EdgeF1 |
| Structural Similarity | 0.12 | exp(−TED/25) |
| Compactness | 0.03 | exp(−\|log((len+50)/(ref+50))\|) |

Score = 0 if SVG fails validity gate (parse error, disallowed tags, >16K chars, >256 paths).

---

## SVG Constraints

- Canvas: 256×256
- Max length: 16,000 characters
- Max paths: 256
- Allowed tags: `svg g path rect circle ellipse line polyline polygon defs use symbol clipPath mask linearGradient radialGradient stop text tspan title desc style pattern marker filter`
- Disallowed: `script`, event handlers (`on*`), animation tags, `foreignObject`, external refs

---

## Reproducibility

- Random seed: 42 (fixed in all code)
- Training: `train_svg.py` + `submit_svg_sft.sh`
- Inference: `kaggle_inference.ipynb`
- Model weights: [to be linked after training]
- GitHub: [this repo]
