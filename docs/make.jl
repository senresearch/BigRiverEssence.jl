using BigRiverEssence
using Documenter

# copy readme into index.md
open(joinpath(@__DIR__, "src", "index.md"), "w") do io
    write(io, read(joinpath(@__DIR__, "..", "README.md"), String))
end

makedocs(; modules=[BigRiverEssence], sitename="BigRiverEssence.jl", pages=[
        "Home" => "index.md",
        "Principal Component Analysis" => "pca_tutorial.md",
        
        "API Reference" => "api.md", 
        # "Example: MLM for ordinal predictors" => "example_ordinal_data.md",
        # "Types and Functions" => "functions.md",
    ]
)

deploydocs(;
    repo = "github.com/BanerjeeAbhisek/BigRiverEssence.jl.git",
    devbranch= "main",
)
