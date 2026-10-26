# Team review guide (Group 2)

Group 2, Trustworthy Machine Learning (Fall 2026): Hanze Li, Sijie Chen, Yuan Zhou, Yiming Liang.
**Submission deadline: Tue Oct 27, 2026, 11:59 PM ET (Canvas); late submissions are not accepted.** Reading the report and checking the list below takes about
15 minutes; no account or installation is needed.

## 1. Read the report

| what | link |
|---|---|
| **Report to submit (PDF, NeurIPS style, 5 pages)** | [reports/paper/Group2_report.pdf](reports/paper/Group2_report.pdf) |
| Same text as a web page | [reports/REPORT.md](reports/REPORT.md) |
| Every experiment, including runs not in the report | [reports/FULL_REPORT.md](reports/FULL_REPORT.md) |

## 2. Check

- [ ] Your name is spelled correctly in the author line, and the order is fine (currently: Hanze Li, Sijie Chen, Yuan Zhou,
      Yiming Liang). If you want your email under your name, send it.
- [ ] The report covers the five required items of the course README, section 8 (see the table below).
- [ ] The Hugging Face pages open without logging in and name the course and Group 2:
  - final model: https://huggingface.co/yuan-zhou-hf/Qwen2.5-3B-Instruct-SFT-advanced15k_ep5
  - random-baseline model: https://huggingface.co/yuan-zhou-hf/Qwen2.5-3B-Instruct-SFT-random15k
  - random 15K dataset: https://huggingface.co/datasets/yuan-zhou-hf/acereason-random15k
  - selected 15K dataset: https://huggingface.co/datasets/yuan-zhou-hf/acereason-advanced15k
- [ ] Optional spot check of numbers: every score in Table 2 comes from
      [reports/five_models/results.md](reports/five_models/results.md) (all models, all benchmarks, bootstrap intervals;
      written as fractions there, so 0.590 is 59.0% in the PDF); the raw per-benchmark logs are in
      [eval_outputs/](eval_outputs/).

**Required items and where they are in the PDF**

| requirement | where in the PDF |
|---|---|
| baseline performance of Qwen2.5-3B-Instruct | Table 2, "Base" columns |
| performance of our fine-tuned model | Table 2, "Final model" columns |
| comparison and analysis | Section 4 |
| hyperparameters of the final model | Table 1 |
| data selection strategy | Section 2 |

## 3. Suggest changes

- **Small fixes** (a word, a name, a sentence): post in the group chat with the section and the sentence, and Yuan will apply
  it and rebuild the PDF.
- **Editing the text yourself:** download [reports/paper/overleaf_upload.zip](reports/paper/overleaf_upload.zip) (the LaTeX
  source, bibliography, NeurIPS style file and figure), then on [Overleaf](https://www.overleaf.com) choose
  *New Project > Upload Project* and upload the zip. It compiles as is with the default pdfLaTeX. Share the Overleaf link in the
  group chat.
- Changes to code or results: open an issue on this repository.

## 4. What we submit on Canvas

1. GitHub repository: https://github.com/yuan-zhou-git/Course_Project_Trustworthy_ML
2. Hugging Face models: the final model and the random-baseline model (links above)
3. Hugging Face datasets: random 15K and selected 15K (links above)
4. Report: [Group2_report.pdf](reports/paper/Group2_report.pdf)

## Where things are

| folder | contents |
|---|---|
| `reports/paper/` | report source (`report.tex`, `refs.bib`), figure script, compiled PDF |
| `reports/five_models/` | final result tables (`results.md`, `results.json`) and truncation analysis |
| `scripts/` | data selection (`prep_data.py`), evaluation aggregation, Hugging Face upload |
| `yamls/`, `slurm/` | training configs and cluster job scripts |
| `eval_outputs/`, `models/` | evaluation logs and training records |
| `docs/progress_log.md` | decision log of the whole project |
