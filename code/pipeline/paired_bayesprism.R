args <- commandArgs(trailingOnly=TRUE);smoke <- '--smoke' %in% args
root <- normalizePath('260925_Re',winslash='/');src <- file.path(root,'runs/paired_deconvolution_inputs_20260926_130450')
out <- file.path(root,'runs',paste0(if(smoke)'paired_bayesprism_smoke_' else 'paired_bayesprism_fixed_',format(Sys.time(),'%Y%m%d_%H%M%S')));dir.create(out,recursive=TRUE)
con <- file(file.path(out,'run.log'),'wt');sink(con,split=TRUE);sink(con,type='message')
suppressPackageStartupMessages({library(BayesPrism);library(Matrix)})
writeLines(c('Measured reference cells and frozen count mixtures from previous paired input run.', 'BayesPrism 2.2.3 initial Gibbs, fixed reference: update.gibbs=FALSE; this is an explicit frozen-reference variant, NOT full reference-updated canonical BayesPrism.', 'outlier.cut=1 prevents any test-cohort-dependent gene filtering; genes selected only external reference donors.',paste('chain.length=',if(smoke)50 else 1000,'burn.in=',if(smoke)25 else 500,'thinning=2','n.cores=2')),file.path(out,'CONFIG.txt'))
for(seed in if(smoke)7 else c(7,17,37)){
 d <- file.path(src,paste0('seed_',seed));o <- file.path(out,paste0('seed_',seed));dir.create(o)
 ref <- as.matrix(readMM(file.path(d,'reference_counts_cells_by_genes.mtx')));lab <- read.csv(file.path(d,'reference_cell_labels.csv'));genes <- read.csv(file.path(d,'gene_ids.csv'));rownames(ref)<-lab$cell_id;colnames(ref)<-genes$gene_id
 mix <- as.matrix(read.csv(file.path(d,'mixture_counts.csv'),row.names=1,check.names=FALSE));if(smoke)mix<-mix[c(1,43),,drop=FALSE]
 stopifnot(all(is.finite(ref)),all(is.finite(mix)),all(mix==round(mix)),all(mix>=0))
 prism <- new.prism(reference=ref,input.type='count.matrix',cell.type.labels=lab$compartment,cell.state.labels=lab$compartment,key='malignant_epithelial',mixture=mix,outlier.cut=1,outlier.fraction=1)
 saveRDS(prism,file.path(o,'prism_input.rds'));started<-Sys.time()
 bp <- run.prism(prism,n.cores=2,update.gibbs=FALSE,gibbs.control=list(chain.length=if(smoke)50 else 1000,burn.in=if(smoke)25 else 500,thinning=2,seed=as.numeric(seed)))
 saveRDS(bp,file.path(o,'BayesPrism.rds'))
 theta <- get.fraction(bp,which.theta='first',state.or.type='state');stopifnot(all(is.finite(theta)),max(abs(rowSums(theta)-1))<1e-6)
 write.csv(theta,file.path(o,'fractions.csv'));write.csv(bp@posterior.initial.cellState@theta.cv,file.path(o,'fractions_posterior_cv.csv'))
 for(k in colnames(theta))write.csv(get.exp(bp,state.or.type='state',cell.name=k),gzfile(file.path(o,paste0('Z_',k,'.csv.gz'))))
 writeLines(capture.output(sessionInfo()),file.path(o,'sessionInfo.txt'));writeLines(paste('elapsed_seconds',as.numeric(Sys.time()-started,units='secs')),file.path(o,'timing.txt'))
 cat('Completed seed',seed,'at',as.character(Sys.time()),'\n')
}
writeLines(if(smoke)'smoke_passed' else 'fixed_reference_BayesPrism_completed',file.path(out,'COMPLETE.txt'));sink(type='message');sink();close(con);cat(out,'\n')
