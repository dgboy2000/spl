prots = {'052','074','108','131','146'};
seeds = {'0000','0001'};
folds = 1:5;
types = 1:4;

objInfo = cell(length(prots),1);

for p = 1:length(prots),
    
    finalObjectives = zeros(length(seeds),length(folds),length(types));

    prot = prots{p};
    for f = 1:length(folds),
        for s = 1:length(seeds),
            seed = seeds{s};
            ret = plotRunInfo(prot,1:4,folds(f),seed,0,0);
            finalObjectives(s,f,:) = ret;
        end
    end


    finalObjectivesBestSeed = zeros(length(folds),length(types));
    for t = 1:length(types),
        for f = 1:length(folds),
            finalObjectivesBestSeed(f,t) = min(finalObjectives(:,f,t));
        end
    end

    meanObjectives = mean(finalObjectivesBestSeed,1);
    stdObjectives = std(finalObjectivesBestSeed,1);

    objInfo{p} = [meanObjectives; stdObjectives];

end