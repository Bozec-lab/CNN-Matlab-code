%% Data sets already labelled
clear
%% Augmentation
imageSourceFolder='C:\Users\nader\OneDrive - University of Toronto\Vitkin lab-Nader-PC\Kester\Raw source images';%Example Folder';

%% Starting task of ROIs subdivision
countLabelledImg=0;%To not double count between labelling sessions (Last labelling sessions is not all inclusive
fullListOfLabelled={''};
for att=1:2%4
% % if att==1||att==3
% %     ToAugment=1;
% % elseif att==2||att==4
% %     ToAugment=0;
% % % end
% if ToAugment==1
%     numRot=4;%rotations, none, 90, 180, 270
%     numRef=2;%reflections, LeftRight, UpDown (not combo to avoid double counting with rotations)
%     numAugs=prod([numRot,numRef]);%prod([numRot,numRef]);
%     
%     imageClassifiedFolder='C:\Users\nader\OneDrive - University of Toronto\Vitkin lab-Nader-PC\Kester\LabelledImagesROIsDataBase\Oct52021_IndMets_splitAugmentedNoUnknownv3'%'C:\Users\nader\OneDrive - University of Toronto\Vitkin lab-Nader-PC\Kester\ImagesROIsDataBase\All together'
%     if ~exist(imageClassifiedFolder,'dir')
%         mkdir(imageClassifiedFolder)
%     end
%     cd(imageClassifiedFolder)
% elseif ToAugment==0
     ToAugment=0
    
    imageClassifiedFolder='C:\Users\nader\OneDrive - University of Toronto\Vitkin lab-Nader-PC\Kester\LabelledImagesROIsDataBase\Oct172021_IndMets_splitNOTAugmentedNoUnknown_V4'%'C:\Users\nader\OneDrive - University of Toronto\Vitkin lab-Nader-PC\Kester\ImagesROIsDataBase\All together'
    if ~exist(imageClassifiedFolder,'dir')
        mkdir(imageClassifiedFolder)
    end
    cd(imageClassifiedFolder)
% end



%%
if att==1 %||att==2
    load('C:\Users\nader\OneDrive - University of Toronto\Vitkin lab-Nader-PC\Kester\Results\labelled images October 16, 2021.mat')%Results September 10, 2021\Labeled images September 10, 2021.mat')
end
%To do % 
if att==2 %||att==4
    load('C:\Users\nader\OneDrive - University of Toronto\Vitkin lab-Nader-PC\Kester\Results\ArchiveJuly172021MoreData\Test run 17July21.mat')%load ground truth file
end
Oldpath1="/Users/kesterng/Desktop/Kester/Example Folder"
Oldpath2="C:\Users\nader\OneDrive - University of Toronto\Vitkin lab-Nader-PC\Kester\Example Folder"
alternativePath = ["C:\Users\nader\OneDrive - University of Toronto\Vitkin lab-Nader-PC\Kester\Raw source images"]%Example Folder"];%Fixing path to directory on computer used to train neural network
%% Correcting image datasource path in groundtruth file
unresolvedPaths = changeFilePaths(gTruth,{[Oldpath1,alternativePath]; [Oldpath2,alternativePath]});%Change recoreded image directory from Kester's computer labelling to where I have them saved
%% Setting Folders
%Next time make label not sublabels in automated function to extact ROIs
%for multi labelling (image labeller) from the start, but for now the idea will be to
%subdivide all images into 100 ROIs each saved as a unique image (count up 
%left to right and up by 10s up to down) and placed with repetition in 4 folders
Classes={'Clarity_of_fibrils','Clarity_of_dbanding','Linear_vs_Kinked_fibrils','Coherence'};
    for ind=1:length(Classes)%creating folders
        if ind==1||ind==2
            SubClasses{ind}={'Clear','NOT Clear'};%,'Unknown'};
        elseif ind==3
            SubClasses{ind}={'Kinked','Linear'};%,'Unknown'};
        elseif ind==4
            SubClasses{ind}={'1','0'};%,'Unknown'};
        end     
    
            for state=1:length(SubClasses{ind})%.(Classes{ind}))
                ClassDirTrain{ind,state}=fullfile(imageClassifiedFolder,'Training Dataset',Classes{ind},[Classes{ind},'___',SubClasses{ind}{state}]);
                ClassDirtest{ind,state}=fullfile(imageClassifiedFolder,'Testing Dataset',Classes{ind},[Classes{ind},'___',SubClasses{ind}{state}]);
                ClassDirAll{ind,state}=fullfile(imageClassifiedFolder,'Full Dataset',Classes{ind},[Classes{ind},'___',SubClasses{ind}{state}]);
                %ClassDir.(Classes{ind}).(SubClasses.(Classes{ind}){state})
                    if ~exist(ClassDirTrain{ind,state},'dir')%~exist(ClassDir.(Classes{ind}).(SubClasses.(Classes{ind}){state}),'dir')
                        mkdir(ClassDirTrain{ind,state})
                    end
                    if ~exist(ClassDirtest{ind,state},'dir')%~exist(ClassDir.(Classes{ind}).(SubClasses.(Classes{ind}){state}),'dir')
                        mkdir(ClassDirtest{ind,state})
                    end
                    if ~exist(ClassDirAll{ind,state},'dir')%~exist(ClassDir.(Classes{ind}).(SubClasses.(Classes{ind}){state}),'dir')
                        mkdir(ClassDirAll{ind,state})
                    end
            end
    end
    
%% Subdividing and saving into folders named by labels
LabelledImgs=[gTruth.LabelData.ROISquares{:,1}];    
    N_ROIs_y=sqrt(size(LabelledImgs,1));
    N_ROIs_x=sqrt(size(LabelledImgs,1));
    I_temp_uncropped=imread(gTruth.DataSource.Source{1});
    I_temp=imcrop(I_temp_uncropped,[2,2,size(I_temp_uncropped,1)-3,size(I_temp_uncropped,2)-3]);%changing to account for black border on images
    AveragescaleofROIs_pix_x=size(I_temp,2)/N_ROIs_x;
    AveragescaleofROIs_pix_y=size(I_temp,1)/N_ROIs_y;
    NumImagesRef=length(gTruth.LabelData.ROISquares)%size(LabelledImgs,2);
    
for imgIndx=1:NumImagesRef%only looks at images for which labels created
    if ~isempty(gTruth.LabelData.ROISquares{imgIndx}) && ~contains([fullListOfLabelled{:}],gTruth.DataSource.Source{imgIndx})
        countLabelledImg=countLabelledImg+1;
        fullListOfLabelled{countLabelledImg}=gTruth.DataSource.Source{imgIndx}
    Img_uncropped=imread(gTruth.DataSource.Source{imgIndx});
    Img=imcrop(Img_uncropped,[2,2,size(Img_uncropped,1)-3,size(Img_uncropped,2)-3]);
    FileNameExtract=split(gTruth.DataSource.Source{imgIndx},'\');
    ROIcount=0;
    for indy=1:N_ROIs_y
        for indx=1:N_ROIs_x
            Rangex=(indx-1)*AveragescaleofROIs_pix_x+(1:AveragescaleofROIs_pix_x);
            Rangey=(indy-1)*AveragescaleofROIs_pix_y+(1:AveragescaleofROIs_pix_y);
                ROIcount=ROIcount+1;
                    ROIconsidered=Img(Rangey,Rangex,:);
                      
                    
                    SplitDetermination=rand(1)%To know assignment of set of transformed ROIs as training or testing
                      
                  if ToAugment==1
                    for th=1:4
                        for FlipOrNot=1:2
                           if (FlipOrNot-1)%sum(Augmentation==[0:3])%if it is 1 to 4
                               ROIconsideredToSave=flip(rot90(ROIconsidered,(th-1)),(FlipOrNot-1));
                           else
                               ROIconsideredToSave=rot90(ROIconsidered,(th-1));
                           end
                            %FileNameROI=[sprintf('ROI-%d (x_%d-%d,y_%d-%d) -Rot%d-Ref%d__',ROIcount,Rangex(1),Rangex(end),Rangey(1),Rangey(end),th,FlipOrNot),FileNameExtract{end}];
                        
                    
                            ROIconsideredFig=figure;
                            ROIconsideredFig.Visible='off';
                            imshow(ROIconsideredToSave);                        
                                FileNameROI=[sprintf('ROI-%d (x_%d-%d,y_%d-%d) -Rot%d-Ref%d__',ROIcount,Rangex(1),Rangex(end),Rangey(1),Rangey(end),th,FlipOrNot),FileNameExtract{end}];%FileNameROI=[sprintf('ROI-%d (x_%d-%d,y_%d-%d)__',ROIcount,Rangex(1),Rangex(end),Rangey(1),Rangey(end)),FileNameExtract{end}];
                        if gTruth.LabelData.ROISquares{imgIndx,1}(ROIcount).AlreadyAnalyzed==1
                            for classIndx=1:length(Classes) %TO OMIT UNKNOWNS!!!
                                if ~isequal(gTruth.LabelData.ROISquares{imgIndx,1}(ROIcount).(Classes{classIndx}),'Unknown')
                                    State=gTruth.LabelData.ROISquares{imgIndx,1}(ROIcount).(Classes{classIndx});
                                        if classIndx==4
                                            State=char(int2str(State));
                                        end
                                                if SplitDetermination>=0.2
                                                    DirectoryClass=fullfile(imageClassifiedFolder,'Training Dataset',Classes{classIndx},[Classes{classIndx},'___',State]);
                                                    DirectoryClass2=fullfile(imageClassifiedFolder,'Full Dataset',Classes{classIndx},[Classes{classIndx},'___',State]);
                                                elseif SplitDetermination<0.2
                                                    DirectoryClass=fullfile(imageClassifiedFolder,'Testing Dataset',Classes{classIndx},[Classes{classIndx},'___',State]);
                                                    DirectoryClass2=fullfile(imageClassifiedFolder,'Full Dataset',Classes{classIndx},[Classes{classIndx},'___',State]);
                                                end
                                            
                                            saveas(ROIconsideredFig,fullfile(DirectoryClass,FileNameROI),'png')
                                            saveas(ROIconsideredFig,fullfile(DirectoryClass2,FileNameROI),'png')
                                end
                            end
                        end
                        end
                    end
                  else %No augmentaion
                      ROIconsideredToSave=ROIconsidered;
                    ROIconsideredFig=figure;
                            ROIconsideredFig.Visible='off';
                            imshow(ROIconsideredToSave);                        
                                FileNameROI=[sprintf('ROI-%d (x_%d-%d,y_%d-%d)__',ROIcount,Rangex(1),Rangex(end),Rangey(1),Rangey(end)),FileNameExtract{end}];%FileNameROI=[sprintf('ROI-%d (x_%d-%d,y_%d-%d)__',ROIcount,Rangex(1),Rangex(end),Rangey(1),Rangey(end)),FileNameExtract{end}];
                        if gTruth.LabelData.ROISquares{imgIndx,1}(ROIcount).AlreadyAnalyzed==1%~isempty(gTruth.LabelData.ROISquares{imgIndx}) && %if gTruth.LabelData.ROISquares{imgIndx,1}(ROIcount).AlreadyAnalyzed==1
                            for classIndx=1:length(Classes) %TO OMIT UNKNOWNS!!!
                                if ~isequal(gTruth.LabelData.ROISquares{imgIndx,1}(ROIcount).(Classes{classIndx}),'Unknown')
                                    State=gTruth.LabelData.ROISquares{imgIndx,1}(ROIcount).(Classes{classIndx});
                                        if classIndx==4
                                            State=char(int2str(State));
                                        end
                                            if SplitDetermination>=0.2
                                                    DirectoryClass=fullfile(imageClassifiedFolder,'Training Dataset',Classes{classIndx},[Classes{classIndx},'___',State]);
                                                DirectoryClass2=fullfile(imageClassifiedFolder,'Full Dataset',Classes{classIndx},[Classes{classIndx},'___',State]);
                                            elseif SplitDetermination<0.2
                                                    DirectoryClass=fullfile(imageClassifiedFolder,'Testing Dataset',Classes{classIndx},[Classes{classIndx},'___',State]);
                                                    DirectoryClass2=fullfile(imageClassifiedFolder,'Full Dataset',Classes{classIndx},[Classes{classIndx},'___',State]);
                                                end %DirectoryClass=fullfile(imageClassifiedFolder,Classes{classIndx},[Classes{classIndx},'___',State]);
                                    saveas(ROIconsideredFig,fullfile(DirectoryClass,FileNameROI),'png')
                                    saveas(ROIconsideredFig,fullfile(DirectoryClass2,FileNameROI),'png')
                                end
                            end
                        end
                  end   
        end
    end
        
    end
end
    
end
            %% Data organization by metric
%            AllROIsAllMetrics=dir(fullfile(imageClassifiedFolder,'**'));
parfor ind1=1:length(Classes) 
    %clearvars AllLabelledROIsDataStores
    AllLabelledROIsDataStores=imageDatastore(fullfile(imageClassifiedFolder,'Full Dataset',Classes{ind1}),'IncludeSubfolders',true,'FileExtensions','.png');
        SaveDataCompiledTab=fullfile(imageClassifiedFolder,sprintf('Labelled data - %s.xlsx',Classes{ind1}))    
            ImageTitles=cell(length(AllLabelledROIsDataStores.Files),1);
            Classification=ImageTitles;
    for ind2=1:length(AllLabelledROIsDataStores.Files)
        ImageTitleTemp2=strsplit(AllLabelledROIsDataStores.Files{ind2},'\');
        ImageTitleTemp=strrep(ImageTitleTemp2{end},'.png','');
            ImageTitles{ind2}=ImageTitleTemp;
        ClassificationTemp2=strsplit(AllLabelledROIsDataStores.Files{ind2},'\');
        ClassificationTemp=strrep(ClassificationTemp2{end-1},sprintf('%s___',Classes{ind1}),'');    
            Classification{ind2}=ClassificationTemp;
    end
%%
CompiledDataTab=table(ImageTitles,Classification,'VariableNames',{'ROI',sprintf('Classification: %s',Classes{ind1})});
writetable(CompiledDataTab,fullfile(SaveDataCompiledTab),'Sheet','AllClassifications','Range',sprintf('A%d:B%d',1,size(CompiledDataTab,1)))%save(fullfile(imagesClassifiedByNetworkFolder,FullclassificationFoldername),
end
%%
            %% Data organization by Condition
%            AllROIsAllMetrics=dir(fullfile(imageClassifiedFolder,'**'));
Classes={'Clarity_of_fibrils','Clarity_of_dbanding','Linear_vs_Kinked_fibrils','Coherence'};
Conditions={'RWS';'CWS';'RDS';'CDS';'radiation wet tx';'control wet tx';'radiation dry tx';'control dry tx';'radiation wet dx';'control wet dx';'radiation dry dx';'control dry dx'};
for ind1=1:length(Classes)%checking over all ROIs within metric class 
        if ind1==1||ind1==2
            SubClasses{ind1}={'Clear','NOT Clear'};%,'Unknown'};
        elseif ind1==3
            SubClasses{ind1}={'Kinked','Linear'};%,'Unknown'};
        elseif ind1==4
            SubClasses{ind1}={'1','0'};%,'Unknown'};
        end
end


Tally_cond_met=zeros(length(Conditions),2,length(SubClasses)); %cell(length(Conditions),length([SubClasses{:}]));

%parfor ind=1:length(AllROIsAllMetrics)%checking over all ROIs
    SaveDataCompiledTab2=fullfile(imageClassifiedFolder,sprintf('All metrics per condition.xlsx'))
    ImageTitles={};%cell([]);
            Classification=ImageTitles;
 for ind1=1:length(Classes)       
    %clearvars AllLabelledROIsDataStores
    AllLabelledROIsDataStores=imageDatastore(fullfile(imageClassifiedFolder,'Full Dataset',Classes{ind1}),'IncludeSubfolders',true,'FileExtensions','.png');
    for ind2=1:length(AllLabelledROIsDataStores.Files)
        ImageTitleTemp2=strsplit(AllLabelledROIsDataStores.Files{ind2},'\');
        ImageTitleTemp=strrep(ImageTitleTemp2{end},'.png','');
            ImageTitles{ind1,ind2}=ImageTitleTemp;
        ClassificationTemp2=strsplit(AllLabelledROIsDataStores.Files{ind2},'\');
        ClassificationTemp=strrep(ClassificationTemp2{end-1},sprintf('%s___',Classes{ind1}),'');    
            Classification{ind1,ind2}=ClassificationTemp;
            for ind3=1:length(Conditions)
                for ind4=1:length(SubClasses{ind1})    
                    if isequal(ClassificationTemp,SubClasses{ind1}{ind4}) && contains(ImageTitleTemp,Conditions{ind3})%Clear is in NOT Clear :-( contains(ClassificationTemp,SubClasses{ind1}{ind4}) && contains(ImageTitleTemp,Conditions{ind3})
                        Tally_cond_met(ind3,ind4,ind1)=Tally_cond_met(ind3,ind4,ind1)+1;
                    end
                end    
            end
    end
 end
 %%
 Tally_cond_metCat=[];
 for ind1=1:length(Classes)
    Tally_cond_metCat=cat(2,Tally_cond_metCat,Tally_cond_met(:,:,ind1));
 end
%%
CompiledDataTab2=table(Conditions,Tally_cond_metCat(:,1),Tally_cond_metCat(:,2),Tally_cond_metCat(:,3),Tally_cond_metCat(:,4),Tally_cond_metCat(:,5),Tally_cond_metCat(:,6),Tally_cond_metCat(:,7),Tally_cond_metCat(:,8),'VariableNames',{'Condition','Clarity_of_fibrils___Clear','Clarity_of_fibrils___NOT Clear','Clarity_of_dbanding___Clear','Clarity_of_dbanding___NOT Clear','Linear_vs_Kinked_fibrils___Kinked','Linear_vs_Kinked_fibrils___Linear','Coherence___1','Coherence___0'});%[SubClasses{:}]});%CompiledDataTab2=table(Conditions,Tally_cond_metCat,'VariableNames',{'Condition','Clear','NOT Clear','Clear','NOT Clear','Kinked','Linear','1','0'});%[SubClasses{:}]});
writetable(CompiledDataTab2,fullfile(SaveDataCompiledTab2),'Sheet','AllClassifications','Range',sprintf('A%d:L%d',1,size(CompiledDataTab2,1)+1))%save(fullfile(imagesClassifiedByNetworkFolder,FullclassificationFoldername),
%do not forget +1 since do not want to omit last row (titles row included)
          %% what was labelled
          count=0;
          full={''};
for att=1:3
%att=1
    if att==1 %||att==2
    load('C:\Users\nader\OneDrive - University of Toronto\Vitkin lab-Nader-PC\Kester\Results\labelled images October 16, 2021.mat')%Results September 10, 2021\Labeled images September 10, 2021.mat')
    end
    %To do % 
    if att==2 %||att==4
    load('C:\Users\nader\OneDrive - University of Toronto\Vitkin lab-Nader-PC\Kester\Results\ArchiveJuly172021MoreData\Test run 17July21.mat')%load ground truth file
    end
    if att==3
    load('C:\Users\nader\OneDrive - University of Toronto\Vitkin lab-Nader-PC\Kester\Results\Results September 10, 2021\Labeled images September 10, 2021.mat')
    end
Oldpath1="/Users/kesterng/Desktop/Kester/Example Folder"
Oldpath2="C:\Users\nader\OneDrive - University of Toronto\Vitkin lab-Nader-PC\Kester\Example Folder"
alternativePath = ["C:\Users\nader\OneDrive - University of Toronto\Vitkin lab-Nader-PC\Kester\Raw source images"]%Example Folder"];%Fixing path to directory on computer used to train neural network
%% Correcting image datasource path in groundtruth file
unresolvedPaths = changeFilePaths(gTruth,{[Oldpath1,alternativePath]; [Oldpath2,alternativePath]});%Change recoreded image directory from Kester's computer labelling to where I have them saved

        for imgIndx=1:length(gTruth.LabelData.ROISquares)%only looks at images for which labels created
            if ~isempty(gTruth.LabelData.ROISquares{imgIndx}) && ~contains([full{:}],gTruth.DataSource.Source{imgIndx})%.Source if add in changeFIlePaths
                count=count+1;
                full{count}=gTruth.DataSource.Source{imgIndx}
            end
        end
end
                                %imageds = imageDatastore(imageClassifiedFolder,'IncludeSubfolders',true,'LabelSource','foldernames');  
