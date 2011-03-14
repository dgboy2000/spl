function [ finalObjectives ] = plotRunInfo( prot, typeRange, fold, seed, showPlot, showIters, getOrder)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    %prots = {'052','074','108','131','146'};
    types = {'','_spl','_newHalf','_newAll'};
    typeNames = {'CCCP','SPL','Uncertainty-Slack','Uncertainty'};
    resultDir = 'results';

    %prot = 4;
    %type = 1;
    %fold = 1;
    %seed = '0000';

    latent = cell(1,numel(typeRange));
    example = cell(1,numel(typeRange));
    hamming = cell(1,numel(typeRange));
    slack = cell(1,numel(typeRange));
    entropy = cell(1,numel(typeRange));
    novelty = cell(1,numel(typeRange));
    obj = cell(1,numel(typeRange));
    numIters = zeros(1,numel(typeRange));
    numEx = 0;
    
    finalObjectives = zeros(1,numel(typeRange));
    
    for t = 1:numel(typeRange),
        type = typeRange(t);
        str = [resultDir '/motif' prot '_' num2str(fold) '_s' seed types{type}];
        objAndTimeLoc = [str '.time'];
        latentLoc = [str '.latent'];
        exampleLoc = [str '.examples'];
        hammingLoc = [str '.hamming'];
        slackLoc = [str '.slack'];
        entropyLoc = [str '.entropy'];
        noveltyLoc = [str '.novelty'];

        latent{t} = load(latentLoc);
        example{t} = load(exampleLoc);
        hamming{t} = load(hammingLoc);
        slack{t} = load(slackLoc);
        entropy{t} = load(entropyLoc);
        novelty{t} = load(noveltyLoc);
        objAndTime = load(objAndTimeLoc);
        obj{t} = objAndTime(:,1);

        numIters(t) = size(example{t},1);
        numEx = size(example{t},2);
        
        finalObjectives(t) = obj{t}(numIters(t));
    end

    if showPlot,
        figure; subplot(2,1,1);
        colors = {'r','b','g','c'};
        hold on;
        for t = 1:numel(typeRange),
            plot(obj{t},colors{typeRange(t)},'LineWidth',3);
            legend({typeNames{typeRange}});
        end
        xlabel('Iteration');
        ylabel('Objective');
        title(['Protein ' prot ', fold ' num2str(fold) ', and seed ' seed]);
        subplot(2,1,2);
        hold on;
        for t = 1:numel(typeRange),
            plot(hamming{t},colors{typeRange(t)},'LineWidth',3);
                legend({typeNames{typeRange}});
        end
        xlabel('Iteration');
        ylabel('Average Hamming Distance');
    end
    
    if showIters,
        for t = 1:numel(typeRange),
            type = typeRange(t);
            relSlack = slack{t};
            relEntropy = entropy{t};
            nIt = numIters(t);
            nRows = ceil(nIt/5);
            nCols = 5;
            hslack = figure;
            hent = figure;
            for i = 1:nIt,
                curSlacks = relSlack(i,1:2500);
                curEntropies = relEntropy(i,1:2500);
                [sortedSlacks indices] = sort(curSlacks);
                figure(hslack);
                subplot(nRows,nCols,i);
                bar(sortedSlacks);
                axis([0 length(sortedSlacks) -1 10]);
                figure(hent);
                subplot(nRows,nCols,i);
                bar(curEntropies(indices));
                axis([0 length(sortedSlacks) 0 10]);
            end
        end

    end
    
    if getOrder,
        for t = 1:numel(typeRange),
            
        end
    end
end

