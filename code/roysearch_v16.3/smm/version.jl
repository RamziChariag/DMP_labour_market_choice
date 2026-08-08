############################################################
# smm/version.jl — the codebase version, defined once
#
# Every entry point prints a version banner BEFORE it includes the solver and SMM
# files, so the constant cannot live in one of those: MCMC_main.jl printing
# smm_main.jl's constant is an undefined reference, since neither includes the
# other. Keeping it here means a release bump touches one line and no banner can
# disagree with another.
############################################################

const ROYSEARCH_VERSION = "16.3"
