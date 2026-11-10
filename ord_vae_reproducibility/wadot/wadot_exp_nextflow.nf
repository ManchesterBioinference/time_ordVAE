nextflow.enable.dsl=2
import groovy.json.JsonOutput


// Build the cross-product: every method x each infile from the matching list
def tuples = []

// Validate config presence
if( !params.methods ) error "params.methods is empty"
if( !params.experiment ) error "params.experiment is empty"
if( !params.input_data ) error "params.input_data is empty"

if( params.reuse_existing == null ) params.reuse_existing = true

// Iterate for method and experiment to create a meta map for each run
params.methods.each { methodName, methodMeta ->
  params.experiment.each { expName, expMeta ->
    def train_columns = expMeta.train_col ?: []
    train_columns.each { train_col_prefix -> 
      
      def lang    = (methodMeta.language ?: 'python') as String
      def isR     = (lang == 'Rscript')

      def allowedExps = methodMeta.allowed_experiments
      if (allowedExps && !(expName in allowedExps)) {
          return // Skips to the next experiment
      }
      
      def meta = methodMeta.clone() // Start with method-specific settings
      meta.method = methodName
      meta.experiment = expName
      // For in_dist, expMeta is the column name. For others, it's an empty string.
      meta.split_col = expMeta.split_col ?: ''
      meta.dataset_base = params.dataset_base

      // Set defaults if not in config (set method specific params in params.methods)
      meta.conda_env = (meta.conda_env ?: (isR ? 'r_RETRO' : 'lightning_2'))

      def allocated_cpus = (meta.cpus ?: ((meta.gpu as boolean) ? 8 : 16)) as int
      meta.cpus = allocated_cpus

      def auto_n_jobs = allocated_cpus > 1 ? allocated_cpus - 1 : 0
      if (auto_n_jobs > 4) auto_n_jobs = 4  // Cap at 4 to avoid over-parallelism

      meta.n_jobs = (meta.n_jobs ?: auto_n_jobs)

      meta.mode = (meta.mode ?: null)  // Use null instead of None
      meta.memory = (meta.memory ?: "${meta.cpus * 8} GB")
      meta.time = (meta.time ?: '08:00:00')
      meta.time_col_prefix = (train_col_prefix ?: '')
      meta.time_col_suffix = (meta.time_col_suffix ?: '')
      
      tuples.add(meta)
    }
  }
}

// Create a channel from the tuples
Channel
  .fromList(tuples)
  .set { RUNS }


process PREPARE_SPLITS {
  tag "prepare_splits"

  publishDir path: "${params.outdir}/data/", mode: 'copy'

  cpus   16
  time   '02:00:00'
  queue  'multicore'

  input: path h5ad_file

  output:
  tuple path(params.processed_h5ad), path(params.processed_seurat), emit: processed_files


  shell:
  '''
    #!/usr/bin/env bash
    set -euo pipefail
    # Activate conda for Python script
    set +u
    if command -v conda >/dev/null 2>&1; then
      eval "$(conda shell.bash hook)"
      conda activate "lightning_2"
    fi
    set -u

    existing_results="!{params.outdir}/data/!{params.processed_h5ad}"
    if [[ "!{params.reuse_existing}" == "true" && -s "${existing_results}" ]]; then
      echo "Reusing existing results: ${existing_results}"
      ln -sfn "${existing_results}" "!{params.processed_h5ad}"
    else
      python !{projectDir}/bin/get_data_splits.py \
        --input_h5 !{h5ad_file} \
        --kfolds !{params.kfold} \
        --out_file !{params.processed_h5ad} \
        !{params.input_meta ? "--meta" + params.input_meta : ''} !{params.test ? '--is_test' : ''}
    fi

    set +u
    # Activate conda for R script
    if command -v conda >/dev/null 2>&1; then
      eval "$(conda shell.bash hook)"
      conda activate "r_RETRO"
    fi
    set -u
    
    existing_seurat="!{params.outdir}/data/!{params.processed_seurat}"
    if [[ "!{params.reuse_existing}" == "true" && -s "${existing_seurat}" ]]; then
      echo "Reusing existing Seurat file: ${existing_seurat}"
      ln -sfn "${existing_seurat}" "!{params.processed_seurat}"
    else
      Rscript !{projectDir}/bin/convert_h5ad_to_seurat.R --input_file !{params.processed_h5ad.replace('.h5ad','_counts_only.h5ad')} \
        --out_file !{params.processed_seurat}
    fi
  '''
}

process RUN_METHOD {
  tag { "${meta.dataset_base}:${meta.method}:${meta.experiment}:${meta.time_col_prefix}" }


  publishDir path: {
     "${params.outdir}/${meta.dataset_base}/${meta.experiment}/${meta.method}/${meta.time_col_prefix}"
  }, mode: 'copy'


  cpus   { meta.cpus }
  memory { meta.memory }
  time   { meta.time }
  queue  { meta.gpu ? 'gpuL' : 'multicore' }
  clusterOptions { meta.gpu ? '-G 1' : '' }  // change to --gres=gpu:1 if your SLURM needs it
  errorStrategy { task ->
    // Retry only for signal-based termination; task can be null during some failure paths.
    def exitCode = task?.exitStatus
    if (exitCode in [137, 143, 140]) {
      return 'retry'
    }
    return 'terminate'
  }
  maxRetries 4

  input:
  tuple val(meta), val(infile)

  output:
  // REQUIRED: at least one results.csv must be produced under the work dir
  tuple val(meta), path("results.csv"), emit: results
  // Optional extras
  tuple val(meta), path("**/metrics.csv"), emit: train_data, optional: true
  tuple val(meta), path("**/*.{png,pdf,tsv,txt}"), emit: artifacts, optional: true
  // Note: checkpoints are written directly to params.outdir for persistence across job restarts

  shell:
  '''
  #!/usr/bin/env bash
  set -euo pipefail

  export OMP_NUM_THREADS=!{meta.cpus}
  export MKL_NUM_THREADS=!{meta.cpus}
  export OPENBLAS_NUM_THREADS=!{meta.cpus}
  export NUMEXPR_NUM_THREADS=!{meta.cpus}
  
  set +u
  # Activate conda
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "!{meta.conda_env}"
  else
    echo "conda not found" >&2
    exit 10
  fi
  
  # Optional per-method PYTHONPATH
  if [[ -n "!{meta.extra_pythonpath}" ]]; then
    export PYTHONPATH="!{meta.extra_pythonpath}:${PYTHONPATH:-}"
  fi
  set -u

  [[ "!{meta.gpu}" == "true" ]] && { command -v nvidia-smi >/dev/null && nvidia-smi || true; }

  if [[ "!{meta.gpu}" == "true" ]]; then
    echo "$CUDA_VISIBLE_DEVICES"
  fi


  echo "Running !{meta.method} (!{meta.language}) on !{infile} [dataset=!{meta.dataset_base} exp=!{meta.experiment} col=!{meta.time_col_prefix}]"

  existing_results="!{params.outdir}/!{meta.dataset_base}/!{meta.experiment}/!{meta.method}/!{meta.time_col_prefix}/results.csv"
  checkpoint_dir="!{params.outdir}/!{meta.dataset_base}/!{meta.experiment}/!{meta.method}/!{meta.time_col_prefix}/checkpoints/"
  
  # Create checkpoint dir in final location (persists even if job is killed)
  mkdir -p "${checkpoint_dir}"
  
  if [[ "!{params.reuse_existing}" == "true" && -s "${existing_results}" ]]; then
    echo "Reusing existing results: ${existing_results}"
    cp "${existing_results}" results.csv
    exit 0
  fi

  # Build arguments
  declare -a ARGS
  ARGS+=("!{projectDir}/!{meta.script}")
  ARGS+=("--infile !{infile}")
  ARGS+=("--experiment" "!{meta.experiment}")
  ARGS+=("--time_col" "!{meta.time_col_prefix}!{meta.time_col_suffix}")
  ARGS+=("--n_cpus" "!{meta.n_jobs}")
  ARGS+=("--checkpoint_dir" "${checkpoint_dir}")

  # Conditionally add arguments for in_dist experiment
  if [[ "!{meta.experiment}" == "in_dist" ]]; then
    ARGS+=("--split_col" "!{meta.split_col}")
  fi

  # Conditionally add mode argument if meta.mode is set (not null)
  if [[ "!{meta.mode}" != "null" && -n "!{meta.mode}" ]]; then
    ARGS+=("--mode" "!{meta.mode}")
  fi


  # Use !{projectDir} to build the correct, absolute path to the script
  case "!{meta.language}" in
    python)
      python ${ARGS[@]}
      ;;
    Rscript|R)
      Rscript ${ARGS[@]}
      ;;
    *)
      echo "Unsupported language: !{meta.language}" >&2
      exit 11
      ;;
  esac
  '''
}

process PREP_ANALYSIS_RECORDS {
  tag { experiment }

  cpus 1
  time '00:10:00'
  queue 'serial'

  input:
    tuple val(experiment), val(records), val(metrics_str)

  output:
    tuple val(experiment), path("records.json"), val(metrics_str)

  script:
  """
  cat > records.json <<'JSON'
${groovy.json.JsonOutput.toJson(records)}
JSON
  """
}


process ANALYSE_GROUP {
  tag { experiment }
  publishDir path: "${params.outdir}/analysis/${experiment}", mode: 'copy'

  cpus 2
  time '01:00:00'
  queue 'multicore'

  input:
    tuple val(experiment), val(records_json), val(metrics_str)

  output:
    // REQUIRED combined output for the experiment
    path "combined_metrics_${experiment}.csv", emit: combined
    path "*_by_col_${experiment}*.pdf"
    path "rank_by_exp_${experiment}*.pdf"
    path "./psuedo_time_dist_plots/", optional: true
    path "./coarse_vs_gt_plots/", optional: true

  shell:
  '''
  #!/usr/bin/env bash
  set -euo pipefail
  
  set +u
  # Activate conda
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "lightning_2"
  else
    echo "conda not found" >&2
    exit 10
  fi
  

  python !{projectDir}/bin/analyse_experiments.py \
    --experiment !{experiment} \
    --records !{records_json} \
    --metrics !{metrics_str} \
    --method_colours '!{groovy.json.JsonOutput.toJson(params.plotting.method_colours)}' \
    --colormap !{params.plotting.colormap} \
    --publishdir "!{params.outdir}/analysis/!{experiment}"
  '''

}

process DATA_SUMMARY_FIG {
  tag "data_summary"

  publishDir path: "${params.outdir}/analysis/data_summary", mode: 'copy'

  cpus 4
  time '01:00:00'
  queue 'multicore'

  input:
    tuple path(h5ad_file), val(columns)

  output:
    path "data_summary.pdf"
    path "window_info.txt"


  shell:
  '''
  #!/usr/bin/env bash
  set -euo pipefail

  set +u
  # Activate conda
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "lightning_2"
  else
    echo "conda not found" >&2
    exit 10
  fi

  python !{projectDir}/bin/data_summary.py \
    --h5ad_file !{h5ad_file} \
    --column_windows !{columns.join(',')} \
    --out_file data_summary.pdf
  '''
}

// Workflow block: wire processes and then access RUN_METHOD.out
workflow {
  // 1. Prepare the single data file with all split columns
  PREPARE_SPLITS(Channel.fromPath(params.input_data))

  // 2. Combine the runs with the correct input file (python vs R)
  RUNS
    .map { meta ->
        def isR = (meta.language == 'Rscript')
        tuple(meta, isR)
    }
    .combine(PREPARE_SPLITS.out.processed_files) // -> [meta, isR, py_file, r_file]
    .map { meta, isR, py_file, r_file ->
        def infile = isR ? r_file : py_file
        tuple(meta, infile)
    }
    .set { RUN_INPUTS }

  // 3. Run all methods
  RUN_METHOD(RUN_INPUTS)

  def EXPERIMENTS_ALL = RUNS.map { m -> m.experiment as String }.distinct()


  // 4. Group results by experiment for downstream analysis
  def RESULTS_BY_EXPERIMENT = RUN_METHOD.out.results
    .map { meta, mfile -> 
        def rec = [
          method: meta.method,
          dataset: meta.dataset_base as String,
          column: meta.time_col_prefix as String,
          column_suffix: meta.time_col_suffix as String,
          results: mfile.toString()
        ]
        tuple(meta.experiment as String, rec)
    }
    .groupTuple()  // -> (experiment, [rec, rec, ...])
    .dump(tag: 'DEBUG_GROUPED_RESULTS')

    // Ensure every experiment triggers analysis, even with zero records
  def ANALYSE_INPUT = EXPERIMENTS_ALL
    .map { exp -> tuple(exp, []) }     // fallback empty list
    .mix(RESULTS_BY_EXPERIMENT)        // add real results if any
    .groupTuple()                      // -> (experiment, [[], [rec...]] or just [[]])
    .map { exp, lists ->         
        // Look up metrics from params.experiment
        def metrics_list = params.experiment[exp]?.metrics ?: []
        // Join them into a comma-separated string for the python script
        def metrics_str = metrics_list.join(',')
        
        tuple(exp, lists.flatten(), metrics_str) 
    }

  ANALYSE_INPUT | PREP_ANALYSIS_RECORDS | ANALYSE_GROUP

  // 5. Generate data summary figure
  def data_summary_cols = params.experiment?.in_dist?.train_col ?: []
  
  PREPARE_SPLITS.out.processed_files
    .map { py_file, r_file -> tuple(py_file, data_summary_cols) }
    | DATA_SUMMARY_FIG

}

workflow analysis_only {
  // If RUNS channel already exists (built from params), reuse it:
  // RUNS emits: tuple( meta, infile )
  RUNS
    .map { meta ->
      // Expected publishDir location for results.csv
      def absResults = file("${workflow.launchDir}/results/wadot/${meta.experiment}/${meta.method}/${meta.time_col_prefix}/results.csv").toString()
      def rec = [
        method: meta.method,
        dataset: meta.dataset_base as String,
        column: meta.time_col_prefix as String,
        column_suffix: meta.time_col_suffix as String,
        results: absResults
      ]
      tuple(meta.experiment as String, rec)
    }
    .groupTuple()                // (experiment, [rec...])
    .map { exp, recs ->
            // We must add the metrics string to match PREP_ANALYSIS_RECORDS input signature
      def metrics_list = params.experiment[exp]?.metrics ?: []
      def metrics_str = metrics_list.join(',')
    tuple(exp, recs, metrics_str)
    
    }
    | PREP_ANALYSIS_RECORDS
    | ANALYSE_GROUP

  // 5. Generate data summary figure
  def data_summary_cols = params.experiment?.in_dist?.train_col ?: []
  def published_h5ad = file("${params.outdir}/data/${params.processed_h5ad}")

  
  DATA_SUMMARY_FIG( 
    tuple( published_h5ad, data_summary_cols ) 
  )
}
