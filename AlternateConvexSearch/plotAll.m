prots = {'052','074','108','131','146'};
seeds = {'0000','0001'};
folds = 1:5;
types = 1:3;

objInfo = cell(length(prots),1);
trainErrorInfo = cell(length(prots),1);
testErrorInfo = cell(length(prots),1);
allTimes = cell(length(prots),1);

for p = 1:length(prots),
    
    finalObjectives = zeros(length(seeds),length(folds),length(types));
    trainErrors = zeros(length(seeds),length(folds),length(types));
    testErrors = zeros(length(seeds),length(folds),length(types));
    times = zeros(length(seeds),length(folds),length(types));
    
    prot = prots{p};
    for f = 1:length(folds),
        for s = 1:length(seeds),
            seed = seeds{s};
            [objs runtimes tr_err tst_err] = plotRunInfo(prot,types,folds(f),seed,0,0,0);
            finalObjectives(s,f,:) = objs;
            trainErrors(s,f,:) = 100*tr_err;
            testErrors(s,f,:) = 100*tst_err;
            times(s,f,:) = runtimes;
        end
    end


    finalObjectivesBestSeed = zeros(length(folds),length(types));
    trainErrorsBestSeed = zeros(length(folds),length(types));
    testErrorsBestSeed = zeros(length(folds),length(types));
    
    for t = 1:length(types),
        for f = 1:length(folds),
            [val ind] = min(finalObjectives(:,f,t));
            finalObjectivesBestSeed(f,t) = min(finalObjectives(:,f,t));
            trainErrorsBestSeed(f,t) = trainErrors(ind,f,t);
            testErrorsBestSeed(f,t) = testErrors(ind,f,t);
        end
    end

    meanObjectives = mean(finalObjectivesBestSeed,1);
    stdObjectives = std(finalObjectivesBestSeed,1);
    
    meanTrainErrors = mean(trainErrorsBestSeed,1);
    stdTrainErrors = std(trainErrorsBestSeed,1);
    
    meanTestErrors = mean(testErrorsBestSeed,1);
    stdTestErrors = std(testErrorsBestSeed,1);

    objInfo{p} = [meanObjectives; stdObjectives];
    trainErrorInfo{p} = [meanTrainErrors; stdTrainErrors];
    testErrorInfo{p} = [meanTestErrors; stdTestErrors];
    allTimes{p} = times;

end