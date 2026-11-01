# T7.2 step 2: write the @meta.data of the GSE202379 Seurat object to TSV.
# Metadata only: no assay, count or expression slot is read or written.
#
# Usage: R_LIBS_USER=~/R_libs Rscript extract_seurat_meta.R <SeuratObject.rds.gz> <out.tsv.gz>
#
# SeuratObject is loaded if it loads; otherwise the slot is read as an attribute, which
# needs no class package (an S4 slot is stored as an attribute of the same name).
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2)
loaded <- suppressWarnings(suppressMessages(requireNamespace("SeuratObject", quietly = TRUE)))
cat("SeuratObject loaded:", loaded, "\n")
# GEO gzipped an already gzip-compressed .rds, so there can be two gzip layers:
# gzfile() removes the outer one, gzcon() the inner one.
probe <- gzfile(args[1], "rb")
magic <- readBin(probe, "raw", 2)
close(probe)
double_gz <- identical(magic, as.raw(c(0x1f, 0x8b)))
cat("double gzip:", double_gz, "\n")
con <- if (double_gz) gzcon(gzfile(args[1], "rb")) else gzfile(args[1], "rb")
obj <- readRDS(con)
close(con)
cat("class:", paste(class(obj), collapse = ","), "package:", attr(class(obj), "package"), "\n")
md <- attr(obj, "meta.data")
stopifnot(is.data.frame(md))
ver <- attr(obj, "version")
cat("object version:", if (is.null(ver)) "NA" else as.character(ver), "\n")
assays <- attr(obj, "assays")
cat("assays:", paste(names(assays), collapse = ","), "\n")
cat("cells:", nrow(md), "columns:", ncol(md), "\n")
out <- data.frame(cell_barcode = rownames(md), md, check.names = FALSE, stringsAsFactors = FALSE)
con <- gzfile(args[2], "w")
write.table(out, con, sep = "\t", quote = FALSE, row.names = FALSE, na = "")
close(con)
for (k in names(md)) {
  v <- md[[k]]
  if (is.factor(v) || is.character(v)) {
    u <- unique(as.character(v))
    if (length(u) <= 60) {
      tb <- sort(table(as.character(v)), decreasing = TRUE)
      cat(k, ":", paste(sprintf("%s=%d", names(tb), as.integer(tb)), collapse = "; "), "\n")
    } else {
      cat(k, ": ", length(u), " distinct values\n", sep = "")
    }
  } else {
    cat(k, ": numeric, range ", paste(signif(range(v, na.rm = TRUE), 4), collapse = " to "), "\n", sep = "")
  }
}
