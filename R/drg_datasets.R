library(readr)
library(dplyr)
library(ggplot2)
library(preprocessCore)

load.matrix <- function(matrix, rowids, colids){
  m = read_tsv(matrix, col_names = FALSE)
  
  r = scan(rowids, what = "character")
  c = scan(colids, what = "character")

  rownames(m) = r
  colnames(m) = c
  
  return(as.matrix(m))
  
}

matrices = list()

matrices[[1]] = load.matrix(
  "scRNASeq_TU_Berlin_Pfizer-selected/Human_PFE_iPSC_Neurons/fpkm_log2_matrix.tsv",
  "scRNASeq_TU_Berlin_Pfizer-selected/Human_PFE_iPSC_Neurons/fpkm_rows.tsv",
  "scRNASeq_TU_Berlin_Pfizer-selected/Human_PFE_iPSC_Neurons/fpkm_cols.tsv"
)

matrices[[2]] = load.matrix(
  "scRNASeq_TU_Berlin_Pfizer-selected/Mouse_JH_DRG_Neurons/fpkm_log2_matrix.tsv",
  "scRNASeq_TU_Berlin_Pfizer-selected/Mouse_JH_DRG_Neurons/fpkm_rows.tsv",
  "scRNASeq_TU_Berlin_Pfizer-selected/Mouse_JH_DRG_Neurons/fpkm_cols.tsv"
)

matrices[[3]] = load.matrix(
  "scRNASeq_TU_Berlin_Pfizer-selected/Mouse_Usoskin_DRG_Neurons/usoskin_log2_matrix.tsv",
  "scRNASeq_TU_Berlin_Pfizer-selected/Mouse_Usoskin_DRG_Neurons/usoskin_rows.tsv",
  "scRNASeq_TU_Berlin_Pfizer-selected/Mouse_Usoskin_DRG_Neurons/usoskin_cols.tsv"
)

matrices[[4]] = load.matrix(
  "scRNASeq_TU_Berlin_Pfizer-selected/Rat_PFE_DRG_Neurons/vst_log2_matrix.tsv",
  "scRNASeq_TU_Berlin_Pfizer-selected/Rat_PFE_DRG_Neurons/vst_rows.tsv",
  "scRNASeq_TU_Berlin_Pfizer-selected/Rat_PFE_DRG_Neurons/vst_cols.tsv"
)

matrices[[5]] = load.matrix(
  "scRNASeq_TU_Berlin_Pfizer-selected/Mouse_Li_DRG_Neurons/fpkm_log2_matrix.tsv",
  "scRNASeq_TU_Berlin_Pfizer-selected/Mouse_Li_DRG_Neurons/fpkm_rows.tsv",
  "scRNASeq_TU_Berlin_Pfizer-selected/Mouse_Li_DRG_Neurons/fpkm_cols.tsv"
)

common.rows = Reduce(intersect, lapply(matrices, rownames))

matrices.filt = lapply(matrices, function(m){
  return(m[common.rows,])
})

merged.matrix = do.call(cbind, matrices.filt)
norm.matrix = normalize.quantiles(merged.matrix)

d = dist(t(norm.matrix)) # euclidean distances between the rows
fit = cmdscale(d, eig=TRUE, k=2) # k is the number of dim

plot.df = as.data.frame(fit$points)

plot.df$Dataset = factor(
  c(
    rep("iPSC", ncol(matrices[[1]])),
    rep("MouseJH", ncol(matrices[[2]])),
    rep("MouseUsoskin", ncol(matrices[[3]])),
    rep("RatPFE", ncol(matrices[[4]])),
    rep("MouseLi", ncol(matrices[[5]]))
  )
)
ggplot(plot.df, aes(x = V1, y = V2, color = Dataset)) + geom_point()