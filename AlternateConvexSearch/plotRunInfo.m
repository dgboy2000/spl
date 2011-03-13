function [ ] = plotRunInfo( prot, typeRange, fold, seed )
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

        numIters(t) = size(example,1);
        numEx = size(example,2);
    end

    figure; subplot(2,1,1);
    colors = {'r','b','g','c'};
    hold on;
    for t = 1:numel(typeRange),
        plot(obj{t},colors{typeRange(t)},'LineWidth',3);
        legend({typeNames{typeRange}});
    end
    xlabel('Iteration');
    ylabel('Objective');
    subplot(2,1,2);
    hold on;
    for t = 1:numel(typeRange),
        plot(hamming{t},colors{typeRange(t)},'LineWidth',3);
            legend({typeNames{typeRange}});
    end
    xlabel('Iteration');
    ylabel('Average Hamming Distance');

end

