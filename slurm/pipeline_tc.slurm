#!/bin/bash
#SBATCH --job-name=rsft_pipe_short
#SBATCH --account=tml_2026
#SBATCH --partition=a100_normal_q
#SBATCH --qos=tc_a100_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --gres=gpu:4
#SBATCH --time=13:00:00
#SBATCH --output=/projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft/logs/%x.%j.out
#SBATCH --error=/projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft/logs/%x.%j.err

# One-allocation pipeline for the short-trace arm on 4 GPUs: preflight -> selection -> SFT -> full evaluation + dev screen
# -> summary. Every phase is idempotent, so resubmitting the same script after a timeout continues where it stopped:
# selection is skipped if short15k.json exists, training resumes from its latest checkpoint (or skips if finished),
# benchmark shards already complete in summary.tsv are skipped, the dev screen skips if its JSON exists.
# Evaluation runs 4 workers (one GPU each) that claim shards from one ordered list (largest first) via atomic mkdir.
set -euo pipefail
P=/projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft
S=/scratch/yuanzhou1/reasoning_sft
RUN=short15k; RUN_ID=short15k_v1; MODEL=$P/models/$RUN
# PIPE_TEST=1: rehearsal of every phase except training (existing advanced15k model, 2 tiny shards, separate output dirs)
TEST=${PIPE_TEST:-0}
if [ "$TEST" = 1 ]; then RUN_ID=pipe_test_v1; MODEL=$P/models/advanced15k; fi
stamp() { echo "=== [$(date '+%F %T')] $*"; }
stamp "job $SLURM_JOB_ID on $(hostname)"
# twin guard: copies of this pipeline may be queued under one job name (normal + preemptable partitions). A copy that
# finds another one RUNNING that started earlier (tie: lower job id) exits. A preempted copy is requeued by Slurm and
# continues from the idempotent phases when it runs again.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  ME=$(squeue -h -j $SLURM_JOB_ID -o "%S" | head -1)
  while read -r OID OST; do
    [ -z "$OID" ] || [ "$OID" = "$SLURM_JOB_ID" ] && continue
    if [[ "$OST" < "$ME" ]] || { [ "$OST" = "$ME" ] && [ "$OID" -lt "$SLURM_JOB_ID" ]; }; then
      stamp "twin $OID running since $OST (this copy started $ME); exiting"; exit 0
    fi
  done < <(squeue -u yuanzhou1 -h -n "$SLURM_JOB_NAME" -t RUNNING -o "%i %S")
fi
nvidia-smi
IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NG=${#G[@]}; echo "GPUs: ${G[*]} (n=$NG)"
# child scripts (train/eval/screen/finalize) call `module`: make the Lmod function visible to child shells, fail fast otherwise
if ! type module >/dev/null 2>&1; then
  set +u; for f in /etc/profile.d/lmod.sh /etc/profile.d/z00_lmod.sh /etc/profile.d/modules.sh; do [ -f $f ] && source $f && break; done; set -u
fi
type module >/dev/null 2>&1 && export -f module
bash -c "type module" >/dev/null 2>&1 || { echo "module command not available to child shells"; exit 6; }

# ---- phase 0: preflight of the eval env (CUDA 13.0 build) on this node; training does not depend on it
# (spawn only for the smoke, as eval_run.slurm sets it itself; NOT exported globally: dev_screen.py has no __main__
#  guard and runs with the default fork method, as on Falcon)
export HF_HOME=$P/hf_cache
EVAL_OK=0
SMF=$P/logs/pipe_smoke_${SLURM_JOB_ID}_${SLURM_RESTART_COUNT:-0}.txt   # full output kept (root causes print far above the last lines)
(set +u; module reset >/dev/null 2>&1; module load Miniconda3/25.11.1-1 CUDA/12.8.0; source activate $P/envs/eval_env; \
 CUDA_VISIBLE_DEVICES=${G[0]} HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn timeout 900 python $P/code/scripts/vllm_smoke.py $P/models/advanced15k 0.5) > $SMF 2>&1 || true
grep -E "^(torch|device|vllm|SMOKE_OK)|Error|error:" $SMF | head -20
grep -q "^SMOKE_OK" $SMF && EVAL_OK=1
stamp "preflight: EVAL_OK=$EVAL_OK"

# ---- phase 1: selection (no GPU work; plain json)
if [ ! -s $P/data/selections/short15k.json ]; then
  stamp "selection"
  (set +u; module reset >/dev/null 2>&1; module load Miniconda3/25.11.1-1; source activate $P/envs/train_env; \
   python $P/code/scripts/select_short.py --pool $P/data/raw/data/acereason11_100k.json --lp-dir $P/data/lp --out $P/data/selections \
     --manifests $P/data/selections/random15k_manifest.json $P/data/selections/advanced15k_manifest.json $P/data/selections/lp15k_manifest.json)
fi
md5sum $P/data/selections/short15k.json
grep -q '"rsft_short15k"' $P/data/selections/dataset_info.json || { echo "rsft_short15k missing from dataset_info.json"; exit 3; }

# ---- phase 2: SFT (same wrapper and recipe as the other arms; resumes / skips by itself)
stamp "training"
set +e
if [ "$TEST" = 1 ]; then echo "(test mode: training skipped)"; RC=0; else RUN=$RUN CONFIG=yamls/short15k.yaml bash $P/slurm/train.slurm; RC=$?; fi
set -e
[ $RC -eq 0 ] || { stamp "training failed (exit $RC)"; exit $RC; }
[ -s $MODEL/config.json ] && ls $MODEL/*.safetensors >/dev/null 2>&1 || { stamp "no exported model in $MODEL"; exit 5; }
stamp "model ready: $MODEL"

if [ $EVAL_OK -ne 1 ]; then
  touch $MODEL/.needs_eval_elsewhere
  stamp "eval env failed preflight on this node: model trained and staged; evaluate it on Falcon (eval_run.slurm). Exiting."
  exit 0
fi

# ---- phase 3: evaluation (19 benchmark shards, same boundaries as the Falcon runs) + dev-panel screen
TASKS=(olympiadbench_0_85 olympiadbench_85_169 olympiadbench_169_254 olympiadbench_254_338 olympiadbench_338_423
       olympiadbench_423_507 olympiadbench_507_591 olympiadbench_591_675 kaoyan_0_100 kaoyan_100_199 minerva_0_136
       minerva_136_272 gpqa_0_99 gpqa_99_198 math_0_125 math_125_250 math_250_375 math_375_500 small screen)
[ "$TEST" = 1 ] && TASKS=(amc_0_2 aime_0_2)
CLAIM=$S/pipe_claims/${SLURM_JOB_ID}_${SLURM_RESTART_COUNT:-0}; mkdir -p $CLAIM   # fresh claims after a preemption requeue
LOGD=$P/logs/pipe_short_$SLURM_JOB_ID; mkdir -p $LOGD
run_task() {   # $1 task, $2 gpu, $3 worker index
  local t=$1
  export CUDA_VISIBLE_DEVICES=$2 SKIP_TWIN_GUARD=1
  case $t in
    screen) MODELS="short_ep3=$MODEL" bash $P/slurm/screen.slurm ;;
    small)  MODEL=$MODEL RUN_ID=$RUN_ID DATASETS="aime amc cn_math_2024" bash $P/slurm/eval_run.slurm ;;
    *)      local d s e; IFS=_ read -r d s e <<< "$t"
            MODEL=$MODEL RUN_ID=$RUN_ID DATASETS=$d START=$s END=$e bash $P/slurm/eval_run.slurm ;;
  esac
}
worker() {     # $1 worker index, $2 gpu
  sleep $(( $1 * 45 ))   # stagger vLLM start-ups (free-port selection, model load from /projects)
  for t in "${TASKS[@]}"; do
    mkdir $CLAIM/$t 2>/dev/null || continue
    stamp "worker $1 (gpu $2): $t start"
    if run_task $t $2 $1 > $LOGD/$t.log 2>&1; then stamp "worker $1: $t done"; else stamp "worker $1: $t FAILED (log $LOGD/$t.log)"; fi
  done
}
stamp "evaluation: ${#TASKS[@]} tasks on $NG GPUs (logs in $LOGD)"
for k in $(seq 0 $((NG - 1))); do worker $k ${G[$k]} & done
wait
stamp "evaluation workers finished"

# ---- phase 4: summary rows + six-model aggregation (cheap; partial runs are reported as such)
set +e
AGG_RUNS="base=base_v1 random=random15k_v1 advanced=advanced15k_v1 ep5=advanced15k_ep5_v1 lp=lp15k_v1 short=short15k_v1"; AGG_OUT=$P/reports/six_models
[ "$TEST" = 1 ] && { AGG_RUNS="advanced=advanced15k_v1 test=pipe_test_v1"; AGG_OUT=$P/reports/pipe_test; }
RUNS="$RUN_ID" AGG_RUNS="$AGG_RUNS" AGG_OUT=$AGG_OUT bash $P/slurm/finalize_summary.slurm > $LOGD/finalize.log 2>&1
set -e
tail -n 30 $LOGD/finalize.log
N_OK=$(awk -F'\t' 'NR>1 && $NF=="True"' $P/eval_outputs/$RUN_ID/summary.tsv 2>/dev/null | wc -l)
stamp "PIPELINE DONE: $N_OK complete summary rows in eval_outputs/$RUN_ID (expected 21: 18 shards + aime, amc, cn_math_2024)"
