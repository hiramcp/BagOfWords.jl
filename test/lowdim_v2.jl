using BagOfWords, TextSearch, MLUtils, StatsBase
using Test, JSON, Downloads

if length(ARGS) < 1
    println("Usage: julia lowdim_v2.jl <experiment_name> <task_name> <train_file> <gold_test_file>")
    exit(1)
end

experiment_name = ARGS[1]
task_name = ARGS[2]
train_file = ARGS[3]
gold_test_file = ARGS[4]
klass_name = ARGS[5]



@testset "lowdim_v2.jl" begin
    datafile = train_file 
    
    data = read_json_dataframe(datafile)

    @info countmap(data.klass)
    n = size(data, 1)
    itrain, itest = splitobs(1:n, at=0.7, shuffle=true)
    #config = (; gw=EntropyWeighting(), lw=BinaryLocalWeighting(), mapfile=nothing, qlist=[2, 5], mindocs=3, collocations=7)
    B = let
        train = data[itrain, :]
        validation = data[itest, :]

        modelselection(train.text, train.klass, 3;
                       projection_options=[RawVectors(), UmapProjection(k=8, layout=RandomLayout(), n_epochs=50, maxoutdim=3)],
                       validation_text=validation.text,
                       validation_labels=validation.klass) do ygold, ypred
            mean(ygold .== ypred)
        end
    end

    for (i, c) in Iterators.reverse(enumerate(B))
        @info i => c.score
        @info i => c.config
    end
    # Write r tests here.
end
