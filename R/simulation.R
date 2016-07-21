set.seed(100)

n.genes     = 1e4
gene.length = 1e4

pop.sizes       = c(250,500,100)
prop.de.per.pop = c(0.1,0.1,0.25)
de.logfc        = c(1, 1, 2)
nde             = prop.de.per.pop * n.genes

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

library(pheatmap)

fpkms = t(t(counts) / (colSums(counts)/1e6)) / gene.length