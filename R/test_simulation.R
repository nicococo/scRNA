library(readr)
library(pheatmap)

source.fname = "source_data_T5_500_S5_500.tsv"
target.fname = "target_data_T5_500_S5_500.tsv"

source.data = read_tsv(source.fname, col_names = FALSE)
target.data = read_tsv(target.fname, col_names = FALSE)

colnames(source.data) = as.character(1:ncol(source.data))
colnames(target.data) = as.character((ncol(source.data)+1):(ncol(source.data)+ncol(target.data)))

source.labels = read_tsv("source_labels_T5_500_S5_500.tsv", col_names = FALSE)
source.labels$X1 = factor(source.labels$X1)
target.labels = read_tsv("target_labels_T5_500_S5_500.tsv", col_names = FALSE)
target.labels$X1 = factor(target.labels$X1)

source.dist.mat = cor(source.data)
rownames(source.labels) = rownames(source.dist.mat)

target.dist.mat = cor(target.data)
rownames(target.labels) = rownames(target.dist.mat)

pheatmap(
  source.dist.mat, 
  annotation_col = source.labels, 
  clustering_method = "ward.D", 
  show_colnames = FALSE, show_rownames = FALSE
)
pheatmap(
  target.dist.mat, 
  annotation_col = target.labels, 
  clustering_method = "ward.D", 
  show_colnames = FALSE, show_rownames = FALSE
)

combined.data = cbind(source.data, target.data)
combined.labels = rbind(source.labels, target.labels)

combined.dist.mat = cor(combined.data)
rownames(combined.labels) = rownames(combined.dist.mat)

combined.labels$X2 = factor(c(rep("Source", ncol(source.data)), rep("Target", ncol(target.data))))

pheatmap(
  combined.dist.mat, 
  annotation_col = combined.labels, 
  clustering_method = "ward.D", 
  show_colnames = FALSE, show_rownames = FALSE
)
