module BigRiverSchneider




using LinearAlgebra, Statistics   

include("pca.jl")
export pca, pcaStructure, pca_transform, pca_invtransform

include("pmd.jl") 
export  pmd, pmd_orth

include("plskern.jl") 
export  plskern, Plsr, plskerncoef, plskernpredict, plskerntransform




end 
