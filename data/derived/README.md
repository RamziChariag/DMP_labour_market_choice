# data/derived — estimation inputs

The SMM entry point (`smm/smm_main.jl`) reads only this directory. These files
are produced by the data pipeline (`data_processing/`, not part of this
solver+smm package) from CPS-ASEC, CPS-basic, JOLTS, J2J, and NSC sources:

    windows.json               window definitions (single source of truth)
    moments_{w}.csv            28 data moments per window (moment, value)
    sigma_{w}.csv              28×28 moment covariance
    nu_estimation.csv          demographic turnover ν (one row per baseline)
    phi_calibration.csv        training completion rate φ
    training_share_scale.csv   κ_w level adjustment for training_share

where {w} ∈ {base_fc, crisis_fc, base_covid, crisis_covid}. Run the data
pipeline to populate this directory before estimating.
