# Clinical covariate diagnostics only; no compartment effects or treatment claims.
suppressPackageStartupMessages(library(survival))
root <- normalizePath('260925_Re', winslash='/')
src <- file.path(root,'runs/clinical_survival_intake_20260925_160636')
out <- file.path(root,'runs',paste0('clinical_cox_diagnostic_',format(Sys.time(),'%Y%m%d_%H%M%S')))
dir.create(out)
writeLines(capture.output(sessionInfo()),file.path(out,'sessionInfo.txt'))
warnings.log <- character()
results <- list();ph.results <- list();checks <- list()
for(cohort in c('TCGA','METABRIC')) {
  d <- read.csv(file.path(src,paste0(cohort,'_patient_survival.csv')),check.names=FALSE)
  stopifnot(!anyDuplicated(d$patient_id),all(d$time_months>0),all(d$event %in% c(0,1)))
  d$age10 <- d$age_years/10
  if(cohort=='TCGA') {
    # Match longest prefix first, preserving pathological stage group.
    x <- toupper(trimws(d$stage));g <- rep(NA_character_,length(x))
    for(k in c('I','II','III','IV')) g[grepl(paste0('^STAGE ',k,'[ABC]?$'),x)] <- k
    d$stage_group <- factor(g,levels=c('I','II','III','IV'))
  } else {
    samples <- read.delim(file.path(root,'data/raw/bulk/metabric/data_clinical_sample.txt'),comment.char='#',check.names=FALSE)
    stopifnot(!anyDuplicated(samples$PATIENT_ID))
    d$TUMOR_STAGE <- samples$TUMOR_STAGE[match(d$patient_id,samples$PATIENT_ID)]
    x <- as.character(d$TUMOR_STAGE);x[!x %in% as.character(0:4)] <- NA
    d$stage_group <- factor(x,levels=c('1','2','3','4','0'))
  }
  for(model in c('age_only','age_stage')) {
    vars <- if(model=='age_only') c('age10') else c('age10','stage_group')
    keep <- complete.cases(d[,vars,drop=FALSE]);dd <- droplevels(d[keep,])
    selection <- d[,c('patient_id','time_months','event','age_years','stage_group')]
    selection$included <- keep;selection$reason <- ifelse(keep,'included','missing_required_covariate')
    write.csv(selection,file.path(out,paste0(cohort,'_',model,'_selection.csv')),row.names=FALSE)
    form <- as.formula(paste('Surv(time_months,event) ~',paste(vars,collapse='+')))
    fit <- withCallingHandlers(coxph(form,data=dd,x=TRUE,y=TRUE,model=TRUE,ties='efron'),
      warning=function(w) {warnings.log <<- c(warnings.log,paste(cohort,model,conditionMessage(w)));invokeRestart('muffleWarning')})
    sm <- summary(fit);co <- data.frame(term=rownames(sm$coefficients),sm$coefficients,sm$conf.int,check.names=FALSE)
    co$cohort <- cohort;co$model <- model;co$n_patients <- nrow(dd);co$events <- sum(dd$event)
    results[[paste(cohort,model)]] <- co
    ph <- cox.zph(fit,transform='km');tab <- as.data.frame(ph$table);tab$term <- rownames(tab);tab$cohort <- cohort;tab$model <- model
    ph.results[[paste(cohort,model)]] <- tab
    saveRDS(list(fit=fit,ph=ph,patient_ids=dd$patient_id),file.path(out,paste0(cohort,'_',model,'_fit.rds')))
    lp <- predict(fit,type='lp');pr <- data.frame(patient_id=dd$patient_id,time_months=dd$time_months,event=dd$event,linear_predictor=lp)
    write.csv(pr,file.path(out,paste0(cohort,'_',model,'_in_sample_linear_predictors.csv')),row.names=FALSE)
    png(file.path(out,paste0(cohort,'_',model,'_schoenfeld.png')),width=1400,height=700,res=140)
    par(mfrow=c(1,length(vars)));plot(ph);dev.off()
    checks[[paste(cohort,model)]] <- data.frame(cohort=cohort,model=model,eligible_survival=nrow(d),complete_covariates=nrow(dd),excluded_missing=sum(!keep),events=sum(dd$event),global_PH_p=ph$table['GLOBAL','p'])
  }
}
write.csv(do.call(rbind,results),file.path(out,'clinical_coefficients.csv'),row.names=FALSE)
write.csv(do.call(rbind,ph.results),file.path(out,'proportional_hazards_tests.csv'),row.names=FALSE)
write.csv(do.call(rbind,checks),file.path(out,'model_cohort_summary.csv'),row.names=FALSE)
writeLines(warnings.log,file.path(out,'model_warnings.txt'))
writeLines(c('Clinical-only diagnostic. All endpoints are overall survival, time in months.',
 'Covariates predeclared: age per 10 years, then age plus categorical stage; no covariate selection using p values.',
 'Complete-case analysis; no imputation. Clinical data are not intersected with expression cohorts.',
 'Proportional-hazards tests are exploratory unadjusted diagnostic tests, not confirmations of clinical biomarkers.',
 'No compartment input, no drug response or treatment-benefit analysis, no validation of original B/plasma HR.',
 'Predictions are in-sample linear predictors for audit only, not out-of-sample survival accuracy.',
 'Source: runs/clinical_survival_intake_20260925_160636; original sources and selection reasons retained there.'),file.path(out,'scope_and_config.txt'))
writeLines('{"status":"completed_clinical_covariate_diagnostics","compartment_survival_validated":false}',file.path(out,'COMPLETE.json'))
print(do.call(rbind,checks));cat(out,'\n')
