# Write a truncated copy of smm_main.jl that stops at a named section marker, so
# the driver's real spec-construction path runs verbatim (same @__DIR__, same
# includes) without entering the optimiser.
#
#   julia slice_spec.jl <path/to/smm_main.jl> <marker> <out.jl>

src_path, marker, out_path = ARGS[1], ARGS[2], ARGS[3]
lines = readlines(src_path)
cut   = findfirst(l -> occursin(marker, l), lines)
cut === nothing && error("marker not found: $marker")
write(out_path, join(lines[1:cut-1], "\n") * "\n")
println("sliced $(cut-1) / $(length(lines)) lines → $out_path")
