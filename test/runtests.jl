using BagOfWords, TextSearch, MLUtils, StatsBase
using Test, JSON, Downloads

@testset "BagOfWords.jl" begin
    datafile = "emo50k.json.gz"
    !isfile(datafile) && Downloads.download("https://github.com/sadit/TextClassificationTutorial/raw/main/data/emo50k.json.gz", datafile)
    data = read_json_dataframe(datafile)
    @info countmap(data.klass)
    data = data[[(c in ("😡", "😍", "😂")) for c in data.klass], :]
    n = size(data, 1)
    itrain, itest = splitobs(1:n, at=0.7, shuffle=true)
    #config = (; gw=EntropyWeighting(), lw=BinaryLocalWeighting(), mapfile=nothing, qlist=[2, 5], mindocs=3, collocations=7)
    B = modelselection(data[itrain, :], 16; validation=data[itest, :]) do ygold, ypred
        mean(ygold .== ypred)
    end

    @info B
    # Write your tests here.
end
