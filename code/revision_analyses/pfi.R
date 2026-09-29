suppressPackageStartupMessages(library(survival))
script_arg<-grep('^--file=',commandArgs(trailingOnly=FALSE),value=TRUE)[1]
script_dir<-dirname(normalizePath(sub('^--file=','',script_arg),mustWork=TRUE))
out<-file.path(script_dir,'clinical');d<-read.csv(file.path(out,'PFI_inputs.csv'),check.names=FALSE)
d$age10<-d$age_years/10;d$stage4<-factor(sub('^Stage (IV|III|II|I).*','\\1',d$stage),levels=c('I','II','III','IV'));d$time_pfi<-d$PFI.time/365.25*12;d$event_pfi<-d$PFI
cc<-complete.cases(d[,c('age10','stage4','time_pfi','event_pfi')]);write.csv(data.frame(patient_id=d$patient_id,in_model=cc),file.path(out,'PFI_complete_case_selection.csv'),row.names=FALSE);d<-d[cc,]
comp<-c('malignant_epithelial','normal_epithelial','CAF','PVL','endothelial','T_NK','B_plasma','myeloid');rows<-list();phs<-list()
for(k in comp){
 d$exposure<-as.numeric(scale(d[[k]]));fit<-coxph(Surv(time_pfi,event_pfi)~exposure+age10+stage4,data=d,ties='efron',x=TRUE);s<-summary(fit);ph<-cox.zph(fit);ci<-s$conf.int['exposure',];rows[[k]]<-data.frame(compartment=k,n=nrow(d),events=sum(d$event_pfi),HR=ci[1],CI_low=ci[3],CI_high=ci[4],p=s$coefficients['exposure','Pr(>|z|)'],PH_exposure=ph$table['exposure','p'],PH_global=ph$table['GLOBAL','p']);saveRDS(fit,file.path(out,paste0('PFI_',k,'.rds')));phs[[k]]<-data.frame(compartment=k,term=rownames(ph$table),ph$table)
}
r<-do.call(rbind,rows);r$BH_q<-p.adjust(r$p,'BH');write.csv(r,file.path(out,'PFI_cox.csv'),row.names=FALSE);write.csv(do.call(rbind,phs),file.path(out,'PFI_PH.csv'),row.names=FALSE);writeLines(capture.output(sessionInfo()),file.path(out,'PFI_session.txt'));print(r)
