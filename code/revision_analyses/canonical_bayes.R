root<-normalizePath('260925_Re',winslash='/');src<-file.path(root,'data/processed/real_count_deconvolution_20260926_142741')
script_arg<-grep('^--file=',commandArgs(trailingOnly=FALSE),value=TRUE)[1]
script_dir<-dirname(normalizePath(sub('^--file=','',script_arg),mustWork=TRUE))
out<-file.path(script_dir,'bayesprism_updated');dir.create(out,recursive=TRUE,showWarnings=FALSE)
suppressPackageStartupMessages({library(BayesPrism);library(Matrix)})
ref<-as.matrix(readMM(file.path(src,'reference_counts_cells_by_genes.mtx')));lab<-read.csv(file.path(src,'reference_cell_labels.csv'));genes<-read.csv(file.path(src,'gene_ids.csv'));rownames(ref)<-lab$cell_id;colnames(ref)<-genes$gene_id
mix<-as.matrix(read.csv(file.path(src,'mixture_counts.csv'),row.names=1,check.names=FALSE));alltheta<-list();allz<-list()
part<-as.integer(commandArgs(trailingOnly=TRUE)[1]);if(is.na(part))part<-0L
for(i in seq_len(nrow(mix))[((seq_len(nrow(mix))-1L)%%4L)==part]){
 id<-rownames(mix)[i];path<-file.path(out,paste0(id,'.rds'))
 if(file.exists(path)){bp<-readRDS(path)}else{
 prism<-new.prism(reference=ref,input.type='count.matrix',cell.type.labels=lab$compartment,cell.state.labels=lab$compartment,key='malignant_epithelial',mixture=mix[i,,drop=FALSE],outlier.cut=1,outlier.fraction=1)
 bp<-run.prism(prism,n.cores=1,update.gibbs=TRUE,gibbs.control=list(chain.length=1000,burn.in=500,thinning=2,seed=7));saveRDS(bp,path)
 }
 theta<-get.fraction(bp,which.theta='final',state.or.type='type');rownames(theta)<-id;alltheta[[id]]<-theta
 for(k in colnames(theta)){v<-matrix(get.exp(bp,state.or.type='type',cell.name=k),nrow=1,dimnames=list(id,genes$gene_id));allz[[k]]<-rbind(allz[[k]],v)}
 cat('COMPLETE SAMPLE',i,id,'\n')
}
write.csv(do.call(rbind,alltheta),file.path(out,paste0('fractions_part',part,'.csv')))
for(k in names(allz))write.csv(allz[[k]],gzfile(file.path(out,paste0('Z_',k,'_part',part,'.csv.gz'))))
writeLines(capture.output(sessionInfo()),file.path(out,paste0('sessionInfo_part',part,'.txt')))
writeLines('Samplewise full update workflow; final type fractions and initial posterior expression per documented get.exp; 1000/500/thin2; seed7; identical 1896 genes and reference; no cross-sample pooling.',file.path(out,'CONFIG.txt'))
writeLines('complete',file.path(out,paste0('COMPLETE_part',part,'.txt')))
