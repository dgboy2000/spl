function [ finalObjectives time trainerror testerror] = plotRunInfo( prot, typeRange, fold, seed, showPlot, showIters, getOrder)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% prot      - string of which protein
% typeRange - vector of which algorithms to look at
% fold, seed- tell which file to read in
% bools: showPlot, showIters, getOrder

    %prots = {'052','074','108','131','146'};
    types = {'','_spl','_unc','_nov'};
    typeNames = {'CCCP','SPL','Uncertainty','Novelty'};
    resultDir = 'results';

    %prot = 4;
    %type = 1;
    %fold = 1;
    %seed = '0000';

    % one cell per alg
    % rows are iterations, columns are examples
    latent = cell(1,numel(typeRange)); % latent variable at each iteration for each point
    example = cell(1,numel(typeRange)); % boolean for was chosen?
    hamming = cell(1,numel(typeRange)); % vec of average hamming dist over selected
    slack = cell(1,numel(typeRange)); % all slacks
    entropy = cell(1,numel(typeRange)); % entropy
    novelty = cell(1,numel(typeRange)); %
    loss = cell(1,numel(typeRange));
    obj = cell(1,numel(typeRange));
    time = zeros(1,numel(typeRange));
    trainerror = zeros(1,numel(typeRange));
    testerror = zeros(1,numel(typeRange));
    numIters = zeros(1,numel(typeRange));
    numEx = cell(1,numel(typeRange));
    runTrainErrors = cell(1,numel(typeRange));
    runTestErrors = cell(1,numel(typeRange));
    
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
        lossLoc = [str '.loss'];
        trainErrorLoc = [str '.trainerror'];
        testErrorLoc = [str '.testerror'];

        latent{t} = load(latentLoc);
        example{t} = load(exampleLoc);
        numEx{t} = sum(example{t},2);
        hamming{t} = load(hammingLoc);
        slack{t} = load(slackLoc);
        entropy{t} = load(entropyLoc);
        novelty{t} = load(noveltyLoc);
        loss{t} = load(lossLoc);
        objAndTime = load(objAndTimeLoc);
        obj{t} = objAndTime(:,1);
        numIters(t) = size(example{t},1);
        time(t) = objAndTime(numIters(t),2);
        trainerror(t) = load(trainErrorLoc);
        testerror(t) = load(testErrorLoc);
        
        if showIters,
            trainTestErrorDir = 'resultsRun';
            str = [trainTestErrorDir '/motif' prot '_' num2str(fold) '_s' seed types{type}];
            trainRunLoc = [str '.trainerrorfile'];
            testRunLoc = [str '.testerrorfile'];
            runTrainErrors{t} = load(trainRunLoc);
            runTestErrors{t} = load(testRunLoc);
        end

        
        finalObjectives(t) = obj{t}(numIters(t));
    end

    if showPlot,
        figure; subplot(1,3,1);
        colors = {'r','b','g','c'};
        hold on;
        for t = 1:numel(typeRange),
            plot(obj{t},colors{typeRange(t)},'LineWidth',3);
            legend({typeNames{typeRange}});
        end
        xlabel('Iteration');
        ylabel('Objective');
        %title(['Protein ' prot ', fold ' num2str(fold) ', and seed ' seed]);
        subplot(1,3,2);
        hold on;
        for t = 1:numel(typeRange),
            plot(hamming{t},colors{typeRange(t)},'LineWidth',3);
                legend({typeNames{typeRange}});
        end
        xlabel('Iteration');
        ylabel('Avg. Hamming Distance');
        subplot(1,3,3);
        hold on;
        for t = 1:numel(typeRange),
            plot(numEx{t},colors{typeRange(t)},'LineWidth',3);
            legend({typeNames{typeRange}});
        end
        xlabel('Iteration');
        ylabel('Number of Examples Used');
        
        if showIters,
            figure; subplot(1,2,1);
            hold on;
            for t = 1:numel(typeRange),
                plot(runTrainErrors{t},colors{typeRange(t)},'LineWidth',3);
                legend({typeNames{typeRange}});
            end
            xlabel('Iteration');
            ylabel('Train Error');
            %title(['Protein ' prot ', fold ' num2str(fold) ', and seed ' seed]);
            subplot(1,2,2);
            hold on;
            for t = 1:numel(typeRange),
                plot(runTestErrors{t},colors{typeRange(t)},'LineWidth',3);
                    legend({typeNames{typeRange}});
            end
            xlabel('Iteration');
            ylabel('Test Error');
        end
    end
    
    if 0,

        
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
                [sortedSlacks indices] = sort(curSlacks);
                figure(hslack);
                subplot(nRows,nCols,i);
                bar(sortedSlacks);
                axis([0 length(sortedSlacks) -1 10]);

                figure(hent);
                subplot(nRows,nCols,i);
                curEntropies = relEntropy(i,1:2500);
                bar(curEntropies(indices));
                axis([0 length(sortedSlacks) 0 10]);
            end
        end
    end
    
    
    if getOrder,
        order = cell(1,numel(typeRange)); % One cell per order vector of entries added for the corresponding algorithm
        spot = cell(1,numel(typeRange)); % Vector: i-th entry is when the i-th point was added
        % {'CCCP','SPL','Uncertainty-Slack','Uncertainty'}
        for t = 1:numel(typeRange),
            ex = example{t};
            
            switch typeNames{typeRange(t)};
              case 'CCCP'
                criteria = zeros(size(latent{t}));
              case 'SPL'
                criteria = loss{t}+slack{t};
              case 'Uncertainty-Slack'
                criteria = (slack{t} + entropy{t}) / 2;
              case 'Uncertainty'
                criteria = loss{t}+entropy{t};
              case 'Novelty'
                criteria = novelty{t};   
              otherwise
                display('Bad typename '+typeNames{t});
                a = 1 / 0;
            end
            
            num_iters = size(ex, 1);
            num_samples = size(ex, 2);
            
            % Take only the first half of the samples with correct label 1
            % This is necessary for the entropy, and therefore also
            % for comparing other things to the entropy
            criteria = criteria(:, 1:(num_samples/2));
            ex = ex(:, 1:(num_samples/2));
            

            [order{t} spot{t}] = ComputeOrder(ex, criteria);
            order{t};
        end
        
        for i = 1:numel(typeRange),
            for j = 1:numel(typeRange),
                if (j <= i)
                    continue
                end
                orderplot = figure;
                figure(orderplot);
                scatter(spot{i}, spot{j}, 4);
                title('Comparison of Orders of Example Addition');
                xlabel(['When example was added by ' typeNames{typeRange(i)} ' algorithm']);
                ylabel(['When example was added by ' typeNames{typeRange(j)} ' algorithm']);
            end
        end
    end
end


function [order, spot, round_samples] = ComputeOrder(examples, criteria)
% Compute the order in which each sample was first added
%  examples - num_iters x num_samples, boolean of whether sample was included in each iteration
%  critera - selection criteria (smaller = more likely to select) for ordering within interations
%
% Order is a vector of the order in which points are added
% Spot is a vector of, for each point, when it is added
  
    num_iters = size(examples, 1);
    num_samples = size(examples, 2);
    
    order = -1000 * ones(1, num_samples);
    spot = -1000*ones(1, num_samples);
    pts_added = 0;

    % Find when each point is first included
    first_rounds = zeros(1, num_samples);
    for j = 1:num_samples,
        first_rounds(j) = find(examples(:,j), 1);
    end
    
    display('starting ComputeOrder')
    for i = 1:num_iters,
        % Find the points first included in this iteration
        round_samples = find(first_rounds == i);
        length(round_samples)
        if (length(round_samples) == 0),
            continue
        end
        
        % Find the permutation of round_samples that orders them by criteria (ascending)
        round_criteria = criteria(i, round_samples);
        [tmp,round_order] = sort(round_criteria);
        
        for j = round_order,
            pts_added = pts_added + 1;
            order(pts_added) = round_samples(j);
            spot(round_samples(j)) = pts_added;
        end
    end
    
end




































