%% Create imageDataStore 
% Define location containing all images
% imageFolder = 'C:\Example Folder\Great Images';
imageAndCodeFolder = 'C:\Users\nader\OneDrive - University of Toronto\Vitkin lab-Nader-PC\Kester' % Change me!
addpath(genpath(imageAndCodeFolder));
imageds = imageDatastore(imageAndCodeFolder,'IncludeSubfolders',true);
%% Load the imageDataStore into the imageLabeller Matlab app which allows you to go through the image and adding classifications
imageLabeler(imageds)
% No need 
% %% Create the data labels and correponding potential binary attributes
ldc = labelDefinitionCreator; 
% addLabel(ldc,'ROISquares',labelType.Rectangle)
%     addAttribute(ldc,'ROISquares','AlreadyAnalyzed',attributeType.Logical,false)
%     addAttribute(ldc,'ROISquares','Clarity_of_fibrils',attributeType.List,{'Clear','NOT Clear','Unknown'})
%     addAttribute(ldc,'ROISquares','Clarity_of_dbanding',attributeType.List,{'Clear','NOT Clear','Unknown'})
%     addAttribute(ldc,'ROISquares','Linear_vs_Kinked_fibrils',attributeType.List,{'Kinked','Linear','Unknown'}) % This can also be made into a continuous numeric range if you "need" (I am sure you do not want!) for defining proportion
%     addAttribute(ldc,'ROISquares','Coherence',attributeType.Logical,true)
% labelDefs=create(ldc)
% save(fullfile(imageAndCodeFolder,'ROISquares_LabelDefs.mat'),'labelDefs','-v7.3') %%Manually loaded in as a label definition from the app
