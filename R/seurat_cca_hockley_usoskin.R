# Seurats CCA on Hockley and Usoskin

# Reading Jims data
data_target = read.table("C:/Users/Bettina/ml/scRNAseq/Data/Jims data/Visceraltpm_m_fltd_mat.tsv");
# reverse log2 for now
data_target = (2^data_target)-1

cell_names_target = read.table("C:/Users/Bettina/ml/scRNAseq/Data/Jims data/Visceraltpm_m_fltd_col.tsv")
transcript_names_target = t(read.table("C:/Users/Bettina/ml/scRNAseq/Data/Jims data/Visceraltpm_m_fltd_row.tsv"))
colnames(data_target)=t(cell_names_target)
rownames(data_target)=transcript_names_target

# Reading Usoskin data
data_source = read.table("C:/Users/Bettina/ml/scRNAseq/Data/Usoskin data/usoskin_m_fltd_mat.tsv");
cell_names_source = read.table("C:/Users/Bettina/ml/scRNAseq/Data/Usoskin data/usoskin_m_fltd_col.tsv")
transcript_names_source = t(read.table("C:/Users/Bettina/ml/scRNAseq/Data/Usoskin data/usoskin_m_fltd_row.tsv"))
colnames(data_source)=t(cell_names_source)
rownames(data_source)=transcript_names_source

# Perform Seurats CCA

library(Seurat)

# Set up target object
target  <- CreateSeuratObject(counts = data_target, project = "Jim",assay = "RNA", min.cells = 18, min.features = 2000)
target$set <- "target"
# target  <- subset(target, subset = nFeature_RNA > 500)
target  <- NormalizeData(target, normalization.method = "LogNormalize")
target  <- FindVariableFeatures(target, selection.method = "vst", nfeatures = 5000)

# Set up stimulated object
source <- CreateSeuratObject(counts = data_source, project = "Usoskin",assay = "RNA", min.cells = 37, min.features = 2000)
source$set <- "source"
# source <- subset(source , subset = nFeature_RNA > 500)
source <- NormalizeData(source, normalization.method = "LogNormalize")
source <- FindVariableFeatures(source , selection.method = "vst", nfeatures = 5000)

immune.anchors <- FindIntegrationAnchors(object.list = list(target, source), dims = 1:20, anchor.features = 5000)
immune.combined <- IntegrateData(anchorset = immune.anchors, dims = 1:20)

# MetageneBicorPlot(immune.combined, grouping.var = "set", dims.eval = 1:20)

# Extract the combined and the two individual datasets
data_comb = immune.combined$integrated
data_target_new = data_comb[,immune.combined$set=="target"]
data_target_new = exp(data_target_new)
data_source_new = data_comb[,immune.combined$set=="source"]
data_source_new = exp(data_source_new)

# Save data
write.table(data_target_new, "C:/Users/Bettina/ml/scRNAseq/Data/Jims data/Jim_after_Seurat.tsv", quote=FALSE, sep='\t', col.names=FALSE,row.names=FALSE)
write.table(colnames(data_target_new), "C:/Users/Bettina/ml/scRNAseq/Data/Jims data/Jim_cell_names_after_Seurat.tsv", quote=FALSE, sep='\t', col.names=FALSE,row.names=FALSE)
write.table(rownames(data_target_new), "C:/Users/Bettina/ml/scRNAseq/Data/Jims data/Jim_gene_names_after_Seurat.tsv", quote=FALSE, sep='\t', col.names=FALSE,row.names=FALSE)


write.table(data_source_new, "C:/Users/Bettina/ml/scRNAseq/Data/Usoskin data/Usoskin_after_Seurat.tsv", quote=FALSE, sep='\t', col.names=FALSE,row.names=FALSE)
write.table(colnames(data_source_new), "C:/Users/Bettina/ml/scRNAseq/Data/Usoskin data/Usoskin_cell_names_after_Seurat.tsv", quote=FALSE, sep='\t', col.names=FALSE,row.names=FALSE)
write.table(rownames(data_source_new), "C:/Users/Bettina/ml/scRNAseq/Data/Usoskin data/Usoskin_gene_names_after_Seurat.tsv", quote=FALSE, sep='\t', col.names=FALSE,row.names=FALSE)

