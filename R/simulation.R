set.seed(100)

n.genes     = 1e4
n.cells     = 1e3
gene.length = 1e4

prop.cells.per.cluster = c(0.25,0.6,0.15) #Must sum to 1
prop.de.per.pop = c(0.1,0.3,0.25) #Must sum to <= 1
de.logfc        = c(2, 1, 2)

nde             = prop.de.per.pop * n.genes
pop.sizes       = prop.cells.per.cluster * n.cells

gamma.shape = 2
gamma.rate  = 2
nb.dispersion = 0.1

true.means = rgamma(n.genes, gamma.shape, gamma.rate)

counts = list()
true.facs = list()

for (x in seq_along(pop.sizes)) { 
  all.facs = 2^rnorm(pop.sizes[x], mean = 0, sd=0.5)
  effective.means = outer(true.means, all.facs, "*")
  
  chosen = c(1, cumsum(nde))[x]:cumsum(nde)[x]
  up.down = sign(rnorm(length(chosen)))
  effective.means[chosen,] = effective.means[chosen,] * 2^(de.logfc[x] * up.down)

  counts[[x]] = matrix(
    rnbinom(n.genes*pop.sizes[x], mu=effective.means, size=1/nb.dispersion), 
    ncol=pop.sizes[x]
  )
}

counts = do.call(cbind, counts)
fpkms = t(t(counts) / (colSums(counts)/1e6)) / gene.length

library(pheatmap)
library(RColorBrewer)
#library(tsne)

cluster.annotation = data.frame(
  Cluster = rep(LETTERS[1:length(pop.sizes)], pop.sizes)
)
cor.df = as.data.frame(cor(log2(fpkms+1)))

rownames(cor.df) = paste("Cell", as.character(1:sum(pop.sizes)), sep = "")
rownames(cluster.annotation) = rownames(cor.df)

pheatmap(
  cor.df, 
  color = colorRampPalette(brewer.pal(n = 7, name = "Blues"))(100),
  annotation_row = cluster.annotation,
  show_rownames = F, show_colnames = F
)