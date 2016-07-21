set.seed(100)

n.genes     = 1e4
gene.length = 1e4
prop.de     = 0.1

pop.sizes   = c(250,500,100)
prop.up.per.pop = c(0.5, 0.5, 0.5)
fc.up = c(5, 5, 5)
fc.down = c(5, 5, 5)

gamma.shape = 2
gamma.rate  = 2
nb.dispersion = 0.1

true.means = rgamma(n.genes, gamma.shape, gamma.rate)
nde = prop.de * n.genes

counts = list()
true.facs = list()

for (x in seq_along(pop.sizes)) { 
  all.facs = 2^rnorm(pop.sizes[x], mean = 0, sd=0.5)
  true.facs[[x]] = all.facs
  effective.means = outer(true.means, all.facs, "*")
  
  chosen = nde * (x-1) + seq_len(nde)
  is.up = seq_len(nde*prop.up.per.pop[x])
  upregulated = chosen[is.up]
  downregulated = chosen[-is.up]
  effective.means[upregulated,] = effective.means[upregulated,] * fc.up[x]
  effective.means[downregulated,] = effective.means[downregulated,] * fc.down[x]
  
  counts[[x]] = matrix(
    rnbinom(ngenes*pop.sizes[x], mu=effective.means, size=1/nb.dispersion), 
    ncol=pop.sizes[x]
  )
}

counts = do.call(cbind, counts)
true.facs = unlist(true.facs)

fpkms = t(t(counts) / (colSums(counts)/1e6)) / gene.length