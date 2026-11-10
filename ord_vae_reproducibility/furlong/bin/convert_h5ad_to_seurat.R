library(Seurat)
library(SeuratDisk)
library(dplyr)
library(optparse)

find_counts_files <- function(base_dir) {
    list.files(base_dir,
        pattern = "*_counts_only\\.h5ad$",
        recursive = TRUE,
        full.names = TRUE
    )
}

process_file <- function(infile, outfile = NULL) {
    cat("Processing:", infile, "\n")

    # Get the base filename without the _counts_only.h5ad suffix
    base_name <- sub("_counts_only\\.h5ad$", "", infile)

    # Define output paths
    h5seurat_file <- sub("\\.h5ad$", ".h5seurat", infile)
    rds_file <- paste0(base_name, "_seurat.rds")

    # Define metadata file paths
    obs_metadata_file <- paste0(base_name, "_obs_metadata.csv")
    var_metadata_file <- paste0(base_name, "_var_metadata.csv")

    tryCatch(
        {
            # Skip if RDS file already exists
            if (file.exists(rds_file)) {
                cat("  RDS file already exists, skipping...\n")
                next
            }
            # Convert h5ad to h5seurat
            cat("  Converting to h5seurat...\n")
            Convert(infile, dest = "h5seurat", overwrite = TRUE)

            # Load the h5seurat file
            cat("  Loading h5seurat file...\n")
            seurat_obj <- LoadH5Seurat(h5seurat_file)

            # Add metadata if files exist
            if (file.exists(obs_metadata_file)) {
                cat("  Adding cell metadata...\n")
                obs_meta <- read.csv(obs_metadata_file, row.names = 1)

                # Match cell names and add metadata
                common_cells <- intersect(rownames(obs_meta), colnames(seurat_obj))
                if (length(common_cells) > 0) {
                    seurat_obj <- AddMetaData(seurat_obj,
                        metadata = obs_meta[common_cells, , drop = FALSE]
                    )
                }
            }

            if (file.exists(var_metadata_file)) {
                cat("  Adding gene metadata...\n")
                var_meta <- read.csv(var_metadata_file, row.names = 1)

                # Add gene metadata to the Seurat object's features
                common_genes <- intersect(rownames(var_meta), rownames(seurat_obj))
                if (length(common_genes) > 0) {
                    seurat_obj[["RNA"]]@meta.features <- var_meta[common_genes, , drop = FALSE]
                }
            }

            if (!is.null(outfile)) {
                rds_file <- outfile
            }
            # Save Seurat object as RDS
            cat("  Saving Seurat object...\n")
            saveRDS(seurat_obj, file = rds_file)

            cat("  Successfully processed:", basename(counts_file), "\n")
        },
        error = function(e) {
            cat("  Error processing", counts_file, ":", e$message, "\n")
        }
    )
}


option_list <- list(
    make_option(c("--input_file"),
        type = "character", default = NULL,
        help = "Input Seurat RDS file", metavar = "character"
    ),
    make_option(c("--input_dir"),
        type = "character", default = NULL,
        help = "Input directory containing h5ad files", metavar = "character"
    ),
    make_option(c("--out_file"),
        type = "character", default = NULL,
        help = "Output Seurat RDS file", metavar = "character"
    )
)
args <- parse_args(OptionParser(option_list = option_list))

if (!is.null(args$input_dir)) {
    cat("Searching for counts files in directory:", args$input_dir, "\n")
    datasets_mirror_dir <- args$input_dir
    counts_files <- find_counts_files(datasets_mirror_dir)

    for (file in counts_files) {
        process_file(file)
        cat("\n")
    }
} else if (!is.null(args$input_file)) {
    cat("Using single input file:", args$input_file, "\n")
    counts_file <- args$input_file
    if (is.null(args$out_file)) {
        stop("When using --input_file, --out_file must also be provided")
    }
    process_file(counts_file, args$out_file)
} else {
    stop("Either --input_dir or --input_file must be provided")
}

cat("Processing complete!\n")
