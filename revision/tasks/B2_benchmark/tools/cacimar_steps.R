# CACIMAR steps of the B2 adapter, called by run_cacimar.py inside cacimar_env.
#   Rscript cacimar_steps.R markers <dir> <species id> <seed>
#   Rscript cacimar_steps.R conserved <dir>
# The tool code runs unmodified; this file only moves data in and out.
args <- commandArgs(trailingOnly = TRUE)
mode <- args[1]
dir <- args[2]
suppressPackageStartupMessages({
  library(Matrix)
  library(Seurat)
  library(CACIMAR)
})

read_lines <- function(f) readLines(file.path(dir, f))

if (mode == "markers") {
  sid <- args[3]
  set.seed(as.integer(args[4]))
  genes <- read_lines(paste0(sid, "_genes.txt"))
  cells <- read_lines(paste0(sid, "_cells.txt"))
  labels <- read_lines(paste0(sid, "_labels.txt"))
  size <- file.size(file.path(dir, paste0(sid, "_x.bin"))) / 8
  i <- readBin(file.path(dir, paste0(sid, "_i.bin")), "integer", size, size = 4)
  p <- readBin(file.path(dir, paste0(sid, "_p.bin")), "integer", length(cells) + 1, size = 4)
  x <- readBin(file.path(dir, paste0(sid, "_x.bin")), "double", size, size = 8)
  counts <- sparseMatrix(i = i + 1L, p = p, x = x, dims = c(length(genes), length(cells)),
                         dimnames = list(genes, cells), repr = "C")
  meta <- data.frame(b2_label = labels, row.names = cells)
  obj <- CreateSeuratObject(counts = counts, meta.data = meta)
  obj <- NormalizeData(obj, verbose = FALSE)
  t0 <- Sys.time()
  m <- Identify_Markers(obj, PowerCutoff = 0.35, DifferenceCutoff = 0.1, cluster_column_use = "b2_label")
  message(sid, " Identify_Markers minutes: ", round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 2))
  saveRDS(m, file.path(dir, paste0(sid, "_markers.rds")))
  out <- m[, c("gene", "Allcluster", "Allpower", "cluster", "power")]
  write.table(out, file.path(dir, paste0(sid, "_markers.tsv")), sep = "\t", quote = FALSE, row.names = FALSE)
} else if (mode == "conserved") {
  hs <- readRDS(file.path(dir, "hs_markers.rds"))
  mm <- readRDS(file.path(dir, "mm_markers.rds"))
  orth <- read.table(file.path(dir, "orthg.tsv"), sep = "\t", header = TRUE, quote = "", comment.char = "",
                     stringsAsFactors = FALSE, check.names = FALSE)
  res <- Identify_ConservedCellTypes(orth, hs, mm, "hs", "mm")
  sct <- res[[2]]
  csct <- sct[grep("^hs", rownames(sct)), grep("^mm", colnames(sct)), drop = FALSE]
  write.table(csct, file.path(dir, "csct.tsv"), sep = "\t", quote = FALSE, col.names = NA)
  pairs <- identify_conserved_pair(csct)
  writeLines(if (is.null(pairs)) character(0) else pairs, file.path(dir, "conserved_pairs.txt"))
  # One call per conserved pair: identify_conserved_gene raises on a pair with a single shared
  # ortholog (the matrix drops to a vector), which must not discard the other pairs.
  all <- list()
  failed <- character(0)
  for (pair in pairs) {
    r <- tryCatch(identify_conserved_marker(orth, hs, mm, "hs", "mm", pair),
                  error = function(e) {
                    failed <<- c(failed, paste0(pair, "\t", conditionMessage(e)))
                    NULL
                  })
    if (!is.null(r)) all[[pair]] <- r
  }
  writeLines(failed, file.path(dir, "conserved_pairs_failed.txt"))
  tab <- if (length(all)) dplyr::bind_rows(all) else
    data.frame(hs_conserved_marker = character(0), mm_conserved_marker = character(0),
               hs_cluster = character(0), mm_cluster = character(0))
  write.table(tab, file.path(dir, "conserved_markers.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
} else {
  stop("unknown mode ", mode)
}
