%clear
%global PosAndDims
PosAndDims=[136,46,78,78];
ToAugment=0%1;%0;--issue is that it may train on an identical image with only a rotation/reflection
%%
Classes={'Clarity_of_fibrils','Clarity_of_dbanding','Linear_vs_Kinked_fibrils','Coherence'};
%% so each rn includes all preceding cumulatively, so combos should all be based on rn=12 taking features from there

for ClassifierNum=1:length(Classes)
%     if Automatic_Rot_Ref
        TrainimageClassifiedFolder=fullfile('C:\Users\nader\OneDrive - University of Toronto\Vitkin lab-Nader-PC\Kester\LabelledImagesROIsDataBase\Oct172021_IndMets_splitNOTAugmentedNoUnknown_V3\Training Dataset',Classes{ClassifierNum})
        TestimageClassifiedFolder=fullfile('C:\Users\nader\OneDrive - University of Toronto\Vitkin lab-Nader-PC\Kester\LabelledImagesROIsDataBase\Oct172021_IndMets_splitNOTAugmentedNoUnknown_V3\Testing Dataset',Classes{ClassifierNum})
%     else
%         imageClassifiedFolder=fullfile('C:\Users\nader\OneDrive - University of Toronto\Vitkin lab-Nader-PC\Kester\LabelledImagesROIsDataBase\October32021_IndividualMetrics_AugmentedNoUnknown',Classes{ClassifierNum})%'C:\Users\nader\OneDrive - University of Toronto\Vitkin lab-Nader-PC\Kester\ImagesROIsDataBase\July172021_All together_Augmented2'
%     end
    
NetworkDir='C:\Users\nader\OneDrive - University of Toronto\Vitkin lab-Nader-PC\Kester\Training Neural Network\Texture classification from Mathworks example';
addpath(genpath('C:\Users\nader\OneDrive - University of Toronto\Vitkin lab-Nader-PC\Kester'));
cd(NetworkDir)

ResultsSaveDir=['C:\Users\nader\Documents\TextureClassifier performance\pre-split not augmented 4 classifiers multi rn combo\',[Classes{ClassifierNum},'Oct132021']];%'C:\Users\nader\OneDrive - University of Toronto\Vitkin lab-Nader-PC\Kester\Training Neural Network\Texture classification from Mathworks example\Training performance\pre-split not augmented 4 classifiers\',[Classes{ClassifierNum},'Oct102021']];
if ~exist(ResultsSaveDir,'dir')
    mkdir(ResultsSaveDir);
end
AccuracyTestFileName=['accuracyTestTableOctober182021NoUnknown_NoAug_combos_',Classes{ClassifierNum},'.mat'];
%% Start parallel pool
if isempty(gcp)
    parpool;
end
%% Defining folder of dataset from which training will occur
%% Already split training and testing data sets
rng(100)%for reproducibity
%rng('default') 
tallrng('default')
Trainds = imageDatastore(TrainimageClassifiedFolder,'IncludeSubfolders',true,'FileExtensions','.png','LabelSource','foldernames');
Testds = imageDatastore(TestimageClassifiedFolder,'IncludeSubfolders',true,'FileExtensions','.png','LabelSource','foldernames');
TrainLabels=Trainds.Labels;
TestLabels=Testds.Labels;
%% Faster to save all Augmented image versions separately into a folder only for training
%and keep another folder for testing (first maybe split directories
%randomly sing the splitlabel function to know how to create the 2 separate
%versions of each folder (now complete)
%% For visual reference
countEachLabel(Trainds)%imageTraindsAug);%imageTraindsAug1.UnderlyingDatastore)%imagedsTrainAug)
countEachLabel(Testds)%imageTestdsAug;%imageTestdsAug1.UnderlyingDatastore)%imagedsTestAug)
%% Create tall arrays for the resized images
Ttrain2= tall(Trainds); %only images not labels adjacent!--do not use transformed datastore I think, apply transfromation in helperScatImages_mean (must be imageDatastore)
    le_Ttrain2=gather(length(Ttrain2));
    
Ttest2 = tall(Testds);
    le_Ttest2=gather(length(Ttest2));
% %% Side test
% imdsReSz = transform(TrainingSet,@(x) rgb2gray(imresize(x,[200,200])));%,'IncludeInfo',true);%[79,79]
% Ttraintesting = tall(imageTraindsAug1);
%% Create a scattering framework for an image input size of 200-by-200 (change to 79 by 79 ? No helperScatImages_mean already resizes to 200 200) with an InvarianceScale of 150. The invariance scale hyperparameter is the only one we set in this example. For the other hyperparameters of the scattering transform, use the default values.
% sn = waveletScattering2('ImageSize',[200,200],'InvarianceScale',150);%must be black and white
%% Preliminary feature extraction
%rn radius of numbering pixels to consider (less is faster)
% rnINI=1
% rnFIN=7%5
rn_used=12%[1:12];%Max is 12
for ind=1:length(rn_used)
        rn=rn_used(ind);%rnINI:rnFIN rn=rnINI:rnFIN
    trainfeatures1 = cellfun(@(img) SSN(img, rn),Ttrain2,'Uni',0);%trainfeatures1{rn}%uniform output false%cellfun(@(x)helperScatImages_meanv2(sn,x,PosAndDims,Automatic_Rot_Ref),Ttrain2,'Uni',0);
    testfeatures1 = cellfun(@(img) SSN(img, rn),Ttest2,'Uni',0);%reshape(cellfun(@(img) SSN(img, rn),Ttest2,'Uni',0),[],size(TrainingSet,2));%cellfun(@(img, radiusPixNeighbourhood) SSN(img, radiusPixNeighbourhood),Ttrain2, rn,'Uni',0);%SSN(Ttest2, rn);%cellfun(@(x)helperScatImages_meanv2(sn,x,PosAndDims, Automatic_Rot_Ref),Ttest2,'Uni',0);
end
%% Using tall's gather capability, gather all the training and test feature vectors and concatenate them into matrices.
%trainfeatures2Resized=reshape(trainfeatures2,[],le_Ttrain2);
Trainf1=gather(trainfeatures1);%cell(length(rn_used(1:end)),1);%rnINI:rnFIN),1);
%     for ind=1:length(rn_used)
%         rn=rn_used(ind);%rnINI:rnFIN
%         Trainf1{rn} =gather(trainfeatures1{rn});%Resized);%reshape(gather(trainfeatures2),[],le_Ttrain2);
%     end
Testf1=gather(testfeatures1);%cell(length(rn_used(1:end)),1);
%     for ind=1:length(rn_used)
%         rn=rn_used(ind);%rnINI:rnFIN
%         Testf1{rn} =gather(testfeatures1{rn});%Resized);%reshape(gather(trainfeatures2),[],le_Ttrain2);
%     end
%%
NumParamPerSSNrnIncrement=length(Trainf1{1})/rn_used(1);%Trainf1{1,1}/rn_used(1);
 %% Loading attempts accuracy achieved
 if exist(fullfile(NetworkDir,AccuracyTestFileName))%'accuracyTestTableSept272021NoUnknown.mat'))
    load(fullfile(NetworkDir,AccuracyTestFileName))%'accuracyTestTableSept272021NoUnknown.mat'))
    rn_used_attempt_count=size(AccuracyTab,1);%rn_NumCount=%Already performed this, now just trying new combinations of neighbourhoods used for analysis
 else
    %AccuracyTab=table([],'rn used', [], 'Accuracy of network');
    rn_used_attempt_count=0%rn_NumCount=0
 end
  %% Organizing data from all feature extraction steps   
  %% Organizing data from all feature extraction steps   
  
for k=1:max(rn_used)%length(rn_used)
  combos=nchoosek([1:rn_used],k);
  for combo=1:size(combos,1)
      
  rn_used_attempt_count=rn_used_attempt_count+1;
  AccuracyTab.rn_Combo{rn_used_attempt_count}=combos(combo,:)
fprintf('Training combo %d based out of %d possible combinations.\n',rn_used_attempt_count,factorial(12)/(factorial(4)*factorial(8)))%2^length(rn_used))
  
% for rn_Num=1:length(rn_used)
%   rn_NumCount=rn_NumCount+1;
%     AccuracyTab.rn_Tested{rn_NumCount}=rn_used(rn_Num);  
% fprintf('Training rn= %d out of %d rn values.\n',rn_NumCount,length(rn_used))

    %trainfeatures0Cat= cat(2,Trainf1{:});
    trainfeatures1Cat=cell(le_Ttrain2,1);
    for samp=1:le_Ttrain2
        for ind=1:length(AccuracyTab.rn_Combo{rn_used_attempt_count})
        rn=AccuracyTab.rn_Combo{rn_used_attempt_count}(ind);%rnINI:rnFIN
            if ind==1           
                trainfeatures1Cat{samp}=[[];(Trainf1{samp}((rn-1)*NumParamPerSSNrnIncrement+(1:NumParamPerSSNrnIncrement)))'];%%[[];trainfeatures0Cat{samp,rn}(:)];%c(:,samp)=[temp;a{samp,rn}(:)];
            else
                trainfeatures1Cat{samp}=[trainfeatures1Cat{samp}(:);(Trainf1{samp}((rn-1)*NumParamPerSSNrnIncrement+(1:NumParamPerSSNrnIncrement)))'];%trainfeatures1Cat{samp}=[trainfeatures1Cat{samp}(:);trainfeatures0Cat{samp,rn}(:)];
            end
        end
    end
    %trainfeatures0Cat=Trainf1{:};%cat(2,Trainf1{rn}); 
    %end
    
    %trainfeatures0Cat= cat(2,Trainf1{:});%2
    trainfeatures2Cat= cat(2,trainfeatures1Cat{:});%cat(2,Trainf1{:});%cat(1,trainfeatures0Cat{:});
%testfeatures2Resized=reshape(testfeatures2,[],le_Ttest2);%le_Ttest2,[]);
    clearvars trainfeatures1Cat
    
    
    %testfeatures0Cat= cat(2,Testf1{:});
    testfeatures1Cat=cell(le_Ttest2,1);
    for samp=1:le_Ttest2
        for ind=1:length(AccuracyTab.rn_Combo{rn_used_attempt_count})
        rn=AccuracyTab.rn_Combo{rn_used_attempt_count}(ind);%rnINI:rnFIN
        if ind==1           
                testfeatures1Cat{samp}=[[];(Testf1{samp}((rn-1)*NumParamPerSSNrnIncrement+(1:NumParamPerSSNrnIncrement)))'];%%[[];trainfeatures0Cat{samp,rn}(:)];%c(:,samp)=[temp;a{samp,rn}(:)];
        else
                testfeatures1Cat{samp}=[testfeatures1Cat{samp}(:);(Testf1{samp}((rn-1)*NumParamPerSSNrnIncrement+(1:NumParamPerSSNrnIncrement)))'];%trainfeatures1Cat{samp}=[trainfeatures1Cat{samp}(:);trainfeatures0Cat{samp,rn}(:)];
        end
%             if ind==1%rn==1           
%                 testfeatures1Cat{samp}=[[];testfeatures0Cat{samp,rn}(:)];%c(:,samp)=[temp;a{samp,rn}(:)];
%             else
%                 testfeatures1Cat{samp}=[testfeatures1Cat{samp}(:);testfeatures0Cat{samp,rn}(:)];
%             end
        end
    end
    %Testf1_1 = gather(testfeatures1);%Resized);%)';%reshape(gather(testfeatures2),[],le_Ttest2);
    %Testf2 =reshape(Testf2_1{:},int8(le_Ttest2),[]);%Testf2_1'; 
    testfeatures2Cat = cat(2,testfeatures1Cat{:});%cat(1,Testf1_1{:})';%2
    clearvars testfeatures1Cat
%     FeaturesTestFileName=sprintf('TestFeatures_rnCombo%d_%s.mat',rn_used_attempt_count,Classes{ClassifierNum});
%     FeaturesTrainFileName=sprintf('TrainFeatures_rnCombo%d_%s.mat',rn_used_attempt_count,Classes{ClassifierNum});
%     
%     save(fullfile(ResultsSaveDir,FeaturesTestFileName),'testfeatures2Cat','-v7.3')
%     save(fullfile(ResultsSaveDir,FeaturesTrainFileName),'trainfeatures2Cat','-v7.3')
%% Train LDA classifier
%With optimization step of LCA classifier for each rn used in SSN for
%metric extraction
% rng('default') 
% tallrng('default')
%,FitInfo,HyperparameterOptimizationResults
[modelLDA] = fitcdiscr(trainfeatures2Cat',TrainLabels,...
    'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('Holdout',0.01,...%0.3%since they used leave one out?
    'AcquisitionFunctionName','expected-improvement-plus'), 'SaveMemory','on')
close all
%% Training PCA classifier
% % for NumDims=101:250
% NumDims=63;
% modelPCA = helperPCAModel(trainfeatures2Cat,NumDims,TrainLabels);%30%carerful need more than 1 example per category%imagedsTrainAug.Labels);%30
% %trainingOptions('sgdm','InitialLearningRate',0.001)%gradient descent solver
%% Evaluate training
%trainingDirectory='C:\Users\nader\OneDrive - University of Toronto\Vitkin lab-Nader-PC\Kester\Training Neural Network\ssn-master\ssn-master'
%mex(fullfile(trainingDirectory,'SSN_getFeatureMaps.cpp'))%trainNetwork
predlabelsLDA=predict(modelLDA,testfeatures2Cat');
%predlabelsPCA = helperPCAClassifier(testfeatures2Cat,modelPCA);

AccuracyTab.accuracy_rn_TestedLDA{rn_used_attempt_count}= sum(TestLabels == predlabelsLDA)./numel(TestLabels)*100 %sum(TestingSet.Labels == predlabels)./numel(TestingSet.Labels)*100
%AccuracyTab.accuracy_rn_TestedPCA{rn_NumCount}= sum(TestLabels == predlabelsPCA)./numel(TestLabels)*100 %sum(TestingSet.Labels == predlabels)./numel(TestingSet.Labels)*100
  end
end
if ~istable(AccuracyTab)
    AccuracyTab=struct2table(AccuracyTab);%table(rn_Tested',accuracy_rn_Tested','VariableNames',{'rn used', 'Accuracy of network'});
end
save(fullfile(ResultsSaveDir,AccuracyTestFileName),'AccuracyTab','-v7.3')
MAXAccuracyLDA(ClassifierNum)=max([AccuracyTab.accuracy_rn_TestedLDA{:}])
%MAXAccuracyPCA(ClassifierNum)=max([AccuracyTab.accuracy_rn_TestedPCA{:}])
%accuracy2(NumDims)
%figure, plotconfusion(TestLabels,predlabels2)
% confusionchart(TestLabels,predlabels2)
% confusionplot(TestLabels,predlabels2)
% end
%%
% function [dataOut,info]=DecolorCropAndAugment(DataStore,info)%,PosAndDims)
%     PosAndDims=[136,46,78,78];
%     numRows = size(DataStore,1);
%     dataOut = cell(numRows,2);
%  
% for idx = 1:numRows
%     %Crop out white and remove color
%     imgOut=rgb2gray(imcrop(DataStore,PosAndDims));%DataStore{idx,1}
%     
%     %Random Rotation
%     if (randi(2)-1)    
%         imgOut=imresize(imrotate(imgOut,rand()*360),[PosAndDims(3)+1,PosAndDims(4)+1]);
%     end
%     
%     if (randi(2)-1) %if 1
%         imgOut=flip(imgOut,randi(2));
%     end
%     dataOut(idx,:)={imgOut,info.Label};%info.Label{idx} imresize(,[200,200])
% end
% end
% %     % Randomized 90 degree rotation
% %     imgOut = rot90(DataStore{idx,1},randi(4)-1);
%     
%     % Return the label from info struct as the 
%     % second column in dataOut.
%     dataOut(idx,:) = {imgOut,info.Label(idx)};
%     
% end
%% Find the best combo
%for i=1:length(AccuracyTab.accuracy_rn_Tested)
ClassifierBest_rnComboLDA{ClassifierNum}=AccuracyTab.rn_Combo{[AccuracyTab.accuracy_rn_TestedLDA{:}]==MAXAccuracyLDA(ClassifierNum)}
%ClassifierBest_rnPCA{ClassifierNum}=find([AccuracyTab.accuracy_rn_TestedPCA{:}]==MAXAccuracyPCA(ClassifierNum))
end
save(fullfile(ResultsSaveDir,'AllClassifiersBestrnCombo_LDA.mat'),'ClassifierBest_rnComboLDA','-v7.3')
%save(fullfile(ResultsSaveDir,'AllClassifiersBest_PCA.mat'),'ClassifierBest_rnPCA','-v7.3')

% image=imread(TrainingSet.Files{1});
% f=SSN(image, rn);
% for ind=1:length(TrainingSet.Files)
%     if mod(ind,50)==0
%         fprintf('%d\n',ind)
%     end
% whichIsSame(ind)=isequal(testfeatures2Cat(:,ind),f);
% end
