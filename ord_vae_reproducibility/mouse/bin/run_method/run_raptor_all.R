#!/usr/bin/env Rscript

# Load required libraries
suppressPackageStartupMessages({
  library(optparse)
  library(Seurat)
  library(SeuratDisk)
  library(SingleCellExperiment)
  library(rlang)
  library(drosoRef)
  library(RAPToR)
  library(biomaRt)
})

run_raptor <- function(seurat_object, ref, time_col, chunk_size = 10000, save_dir = "./") {
  # Get total number of cells
  total_cells <- ncol(seurat_object)
  cell_names <- colnames(seurat_object)

  # Calculate number of chunks needed
  n_chunks <- ceiling(total_cells / chunk_size)
  all_predictions <- list()

  cat("Processing", total_cells, "cells in", n_chunks, "chunks of size", chunk_size, "\n")

  # Process each chunk
  for (i in 1:n_chunks) {
    # Calculate start and end indices for this chunk
    start_idx <- (i - 1) * chunk_size + 1
    end_idx <- min(i * chunk_size, total_cells)

    cat("Processing chunk", i, "of", n_chunks, "(cells", start_idx, "to", end_idx, ")\n")
    gc()

    # Extract chunk of cells
    chunk_cells <- cell_names[start_idx:end_idx]
    chunk_seurat <- seurat_object[, chunk_cells]

    # Convert to SingleCellExperiment
    sce <- as.SingleCellExperiment(chunk_seurat)
    expr <- as.matrix(counts(sce)) # or logcounts(sce), depending on what you want


    cat("running raptor ae")

    # Run RAPToR prediction
    pred <- RAPToR::ae(expr, ref)


    # Write predictions to the output directory
    # Create a dataframe for the current chunk's predictions
    chunk_df <- data.frame(
      cell_id = rownames(pred$age.estimates),
      predicted_age = pred$age.estimates[, 1],
      gt_window = chunk_seurat@meta.data[chunk_cells, args$time_col],
      split = "all_data"
    )
    all_predictions[[i]] <- chunk_df
    cat("Finished processing chunk", i, "\n")
  }
  final_results_df <- do.call(rbind, all_predictions)
  return(final_results_df)
}


# Define command line options
option_list <- list(
  make_option(c("--infile"),
    type = "character", default = NULL,
    help = "Input Seurat RDS file", metavar = "character"
  ),
  make_option(c("--out_file"),
    type = "character", default = "results.csv",
    help = "Output Seurat RDS file", metavar = "character"
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
  make_option(c("--n_cpus"),
    type = "integer", default = 1,
    help = "Number of CPUs to use [default= %default]", metavar = "integer"
  ),
  make_option(c("--split_col"),
    type = "character", default = NULL,
    help = "Column name to split data by (e.g., batch) [default= %default]", metavar = "character"
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

# Validate that all input files exist
if (!file.exists(args$infile)) {
  cat("Error: Input file does not exist:", args$infile, "\n")
  quit(status = 1)
}


# Print parsed arguments if verbose
if (args$verbose) {
  cat("Parsed arguments:\n")
  cat("  Input files:", paste(infile, collapse = ", "), "\n")
  cat("  Time column:", args$time_col, "\n")
  cat("  Verbose:", args$verbose, "\n")
}

# Main analysis code would go here
cat("Running raptor analysis\n")

concat_seurat <- readRDS(args$infile)

# Normalize and log-transform
concat_seurat <- NormalizeData(concat_seurat, normalization.method = "LogNormalize", scale.factor = 10000)

ref <- prepare_refdata(
  datapkg = "drosoRef",
  ref = "Dme_embryo",
  600
)

# Connect to FlyBase BioMart
mart <- useMart("ENSEMBL_MART_ENSEMBL", dataset = "dmelanogaster_gene_ensembl")

# Map gene symbols -> FlyBase IDs
mapping <- getBM(
  attributes = c("external_gene_name", "flybase_gene_id"),
  filters = "external_gene_name",
  values = rownames(concat_seurat),
  mart = mart
)


gene_map <- setNames(mapping$flybase_gene_id, mapping$external_gene_name)
current_genes <- rownames(concat_seurat)
new_genes <- ifelse(current_genes %in% names(gene_map),
  gene_map[current_genes],
  current_genes
)

# Rename genes (features) instead of cells
# Rename genes (features) in the RNA assay
DefaultAssay(concat_seurat) <- "RNA"
rownames(concat_seurat[["RNA"]]@counts) <- new_genes
rownames(concat_seurat[["RNA"]]@data) <- new_genes
if (!is.null(concat_seurat[["RNA"]]@scale.data) && nrow(concat_seurat[["RNA"]]@scale.data) > 0) {
  rownames(concat_seurat[["RNA"]]@scale.data) <- new_genes
}



final_predictions <- run_raptor(
  seurat_object = concat_seurat, ref = ref,
  time_col = args$time_col, chunk_size = args$chunk_size
)

write.csv(final_predictions, args$out_file, row.names = FALSE)
cat("Successfully saved all predictions to", args$out_file, "\n")
