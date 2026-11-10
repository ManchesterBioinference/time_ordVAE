# Load required libraries
suppressPackageStartupMessages({
  library(optparse)
  library(Biobase) # data processing + extraction
  library(ggplot2) # plotting
  library(patchwork) # plotting
  library(doParallel)
  library(parallel)
  library(Seurat)
  library(SeuratDisk)
  library(SingleCellExperiment)
  library(RETRO)
  library(rlang)
})

run_retro_chunked <- function(concat_seurat, time_col, pca_embeddings, pca_loadings, chunk_size = 10000, save_dir = "./") {
  total_cells <- ncol(concat_seurat)
  n_chunks <- ceiling(total_cells / chunk_size)

  # Get variable features once
  var_features <- VariableFeatures(concat_seurat)

  # Random sampling without replacement
  shuffled_indices <- sample(1:total_cells, total_cells, replace = FALSE)

  fails <- 0
  all_predictions <- list()

  for (chunk in 1:n_chunks) {
    cat("Processing chunk", chunk, "of", n_chunks, "\n")
    gc()

    if (n_chunks == 1) {
      cat("Only one chunk detected; processing all data at once.\n")
      chunk_seurat <- concat_seurat[var_features, ]
      chunk_pca <- pca_embeddings # Use all cells
    } else {
      start_idx <- (chunk - 1) * chunk_size + 1
      end_idx <- min(chunk * chunk_size, total_cells)
      cell_indices <- shuffled_indices[start_idx:end_idx]
      cell_names <- colnames(concat_seurat)[cell_indices]

      # Subset Seurat object for the current chunk - genes first, then cells
      chunk_seurat <- concat_seurat[var_features, cell_names]
      chunk_pca <- pca_embeddings[cell_names, ] # Use cell names for indexing
    }

    # Add this right after the chunk_seurat line to debug:
    cat("Chunk seurat dimensions:", dim(chunk_seurat), "\n")
    cat("Expression data dimensions:", dim(GetAssayData(chunk_seurat, slot = "data")), "\n")
    cat("PCA dimensions:", dim(chunk_pca), "\n")
    cat("Variable features length:", length(var_features), "\n") # Add this right after the chunk_seurat line to debug:
    cat("Chunk seurat dimensions:", dim(chunk_seurat), "\n")
    cat("Expression data dimensions:", dim(GetAssayData(chunk_seurat, slot = "data")), "\n")
    cat("PCA dimensions:", dim(chunk_pca), "\n")
    cat("Variable features length:", length(var_features), "\n")



    if (packageVersion("Seurat") >= "5.0.0") {
      chunk_seurat <- JoinLayers(chunk_seurat)
    }

    # added due to bug in as.SingleCellExperiment
    chunk_seurat@assays$RNA@scale.data <- as.matrix(0)
    sce_chunk <- as.SingleCellExperiment(chunk_seurat)


    # Add PCA embeddings
    reducedDim(sce_chunk, "PCA") <- chunk_pca

    # Convert to ExpressionSet
    eset <- ExpressionSet(
      assayData = as.matrix(assay(sce_chunk)),
      phenoData = AnnotatedDataFrame(as.data.frame(colData(sce_chunk)))
    )

    # Create a proper prcomp-like object for RETRO
    chunk_prcomp <- list(
      x = chunk_pca,
      sdev = concat_seurat[["pca"]]@stdev,
      rotation = pca_loadings,
      center = FALSE,
      scale = FALSE
    )

    # Make it look like a prcomp object
    class(chunk_prcomp) <- "prcomp"

    # Add PCA info
    experimentData(eset)@other[["PCA"]] <- chunk_prcomp
    pData(eset)$time <- as.factor(pData(eset)[[time_col]])
    time <- eset@phenoData$time
    cat(sort(unique(time)), "\n")
    time <- as.numeric(levels(time))[time]

    # Run RETRO analysis
    tryCatch(
      {
        lambda <- 1
        coordinates <- weight_coord(eset, weight = lambda)
        coordinates_uw <- weight_coord(eset, weight = 0)

        k_range <- seq(9, 14)
        num_scores <- 20

        retro_obj <- set_RETRO_class(
          coordinates = coordinates,
          coordinates_uw = coordinates_uw,
          k_range = k_range,
          time = time
        )

        if (!interactive()) {
          foreach::registerDoSEQ()
        } else {
          ncores <- parallel::detectCores()
          registerDoParallel(cores = ncores)
        }
        cat("scoring\n")
        retro_obj <- scoring(retro_obj, k_range, num_scores)
        cat("getting lineages\n")
        retro_obj <- get_num_lineages(retro_obj, percent = 0.2)
        cat("getting bezier curves\n")
        retro_obj <- get_bezier_curve(retro_obj)
        cat("fitting psuedotime\n")
        retro_obj <- pseudotime_fit(retro_obj)

        # Extract results
        chunk_meta <- pData(eset)
        chunk_meta$pseudotime <- retro_obj@pseudotime
        chunk_meta$chunk <- chunk

        chunk_to_save <- data.frame(
          cell_id = rownames(chunk_meta),
          pseudotime = chunk_meta$pseudotime,
          gt_window = chunk_meta[[time_col]],
          chunk = chunk_meta$chunk,
          split = "all_data"
        )

        all_predictions[[chunk]] <- chunk_to_save
        cat("Finished processing chunk", chunk, "\n")
      },
      error = function(e) {
        cat("Error in chunk", chunk, ":", e$message, "\n")
        fails <<- fails + 1
        cat("Total failed chunks so far:", fails, "\n")
        if (fails >= 10) {
          cat("Too many failed chunks. Exiting.\n")
          stop("Exiting due to multiple chunk failures.")
        }
      }
    )
  }
  final_results_df <- do.call(rbind, all_predictions)
  cat("finished all chunks\n")
  cat("Total failed chunks:", fails, "\n")
  return(final_results_df)
}

# Define command line options
option_list <- list(
  make_option(c("--infile"),
    type = "character", default = NULL,
    help = "Input Seurat RDS file", metavar = "character"
  ),
  make_option(c("--time_col"),
    type = "character", default = "time",
    help = "Column name for time variable [default= %default]", metavar = "character"
  ),
  make_option(c("--chunk_size"),
    type = "integer", default = 100000,
    help = "Number of cells per chunk [default= %default]", metavar = "integer"
  ),
  make_option(c("--experiment"),
    type = "character", default = NULL,
    help = "Name of the experiment [default= %default]", metavar = "character"
  ),
  make_option(c("--split_col"),
    type = "character", default = NULL,
    help = "Column name to split data by (e.g., batch) [default= %default]", metavar = "character"
  ),
  make_option(c("--n_cpus"),
    type = "integer", default = 1,
    help = "Number of CPUs to use [default= %default]", metavar = "integer"
  ),
  make_option(c("--verbose"),
    action = "store_true", default = FALSE,
    help = "Enable verbose output [default= %default]"
  )
)

# Parse arguments
parser <- OptionParser(
  option_list = option_list,
  description = "Run retro analysis with specified parameters",
  usage = "%prog [options]"
)
args <- parse_args(parser)

# Check required arguments
if (is.null(args$infile)) {
  cat("Error: infile is required\n")
  print_help(parser)
  quit(status = 1)
}

# Parse input files (handle comma-separated list)
input_files <- trimws(unlist(strsplit(args$infile, ",")))

# Validate that all input files exist
for (file in input_files) {
  if (!file.exists(file)) {
    cat("Error: Input file does not exist:", file, "\n")
    quit(status = 1)
  }
}

# Parse input files (handle comma-separated list)

# Validate that all input files exist
if (!file.exists(args$infile)) {
  cat("Error: Input file does not exist:", args$infile, "\n")
  quit(status = 1)
}


# Print parsed arguments if verbose
if (args$verbose) {
  cat("Parsed arguments:\n")
  cat("  Input files:", paste(input_file, collapse = ", "), "\n")
  cat("  Time column:", args$time_col, "\n")
  cat("  Verbose:", args$verbose, "\n")
}

# Main analysis code would go here
cat("Running raptor analysis\n")

concat_seurat <- readRDS(args$infile)

# Find variable features and normalize
concat_seurat <- FindVariableFeatures(concat_seurat, selection.method = "vst", nfeatures = 2000)
concat_seurat <- NormalizeData(concat_seurat, normalization.method = "LogNormalize", scale.factor = 10000)

# Scale data before PCA
concat_seurat <- ScaleData(concat_seurat, features = VariableFeatures(concat_seurat))

# Use Seurat's PCA (works on sparse matrices)
concat_seurat <- RunPCA(concat_seurat, features = VariableFeatures(concat_seurat), npcs = 50)

# Extract PCA embeddings for use in chunks
pca_embeddings <- Embeddings(concat_seurat, reduction = "pca") # cells x PCs
pca_loadings <- Loadings(concat_seurat, reduction = "pca") # genes x PCs

final_predictions <- run_retro_chunked(concat_seurat,
  time_col = args$time_col, pca_embeddings = pca_embeddings,
  pca_loadings = pca_loadings, chunk_size = args$chunk_size
)

write.csv(final_predictions, "results.csv", row.names = FALSE)
cat("Successfully saved predictions to results.csv\n")
