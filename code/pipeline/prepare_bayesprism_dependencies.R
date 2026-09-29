# All libraries and caches are local; no update of system packages.
root <- normalizePath('260925_Re', winslash='/')
lib <- file.path(root,'runtime/r-library')
cache <- file.path(root,'runtime/installers/r-packages')
dir.create(cache, recursive=TRUE, showWarnings=FALSE)
options(repos=c(CRAN='https://mirrors.tuna.tsinghua.edu.cn/CRAN/'), timeout=90,
        Ncpus=2, download.file.method='libcurl')
if (!requireNamespace('BiocManager',quietly=TRUE)) {
  install.packages('BiocManager',lib=lib,type='binary',destdir=cache)
}
stopifnot(requireNamespace('BiocManager',quietly=TRUE))
repos <- BiocManager::repositories(version='3.22')
# Fixed R-compatible Bioconductor release. Each downloaded package is preserved.
options(repos=repos)
needed <- c('snowfall','NMF','gplots','scran','BiocParallel','Matrix')
missing <- needed[!vapply(needed,requireNamespace,logical(1),quietly=TRUE)]
if(length(missing)) install.packages(missing,lib=lib,type='binary',destdir=cache)
ip <- installed.packages()
write.csv(ip[,c('Package','Version','LibPath','Built')],file.path(root,'runtime/R_installed_packages.csv'),row.names=FALSE)
writeLines(capture.output(sessionInfo()),file.path(root,'runtime/R_session_info.txt'))
status <- data.frame(package=needed,available=vapply(needed,requireNamespace,logical(1),quietly=TRUE))
write.csv(status,file.path(root,'runtime/BayesPrism_dependency_status.csv'),row.names=FALSE)
print(status)
if(!all(status$available)) quit(status=2)
