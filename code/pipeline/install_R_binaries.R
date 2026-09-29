args <- commandArgs(trailingOnly=TRUE)
root <- normalizePath('260925_Re',winslash='/')
lib <- file.path(root,'runtime/r-library'); dir.create(lib,showWarnings=FALSE,recursive=TRUE)
pkgs <- list.files(file.path(root,'runtime/installers/r-packages'),pattern='[.]zip$',full.names=TRUE)
sink(file.path(root,'runtime/installers/R_binary_install.log'),split=TRUE)
for (pkg in pkgs) {
 name <- sub('_[^_]*[.]zip$','',basename(pkg))
 if (!dir.exists(file.path(lib,name))) install.packages(pkg,lib=lib,repos=NULL,type='win.binary')
}
print(installed.packages(lib.loc=lib)[,c('Package','Version')]);sink()
