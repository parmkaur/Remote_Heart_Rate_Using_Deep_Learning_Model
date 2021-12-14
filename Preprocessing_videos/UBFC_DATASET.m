% Simple code to read ground truth of the UBFC_DATASET
% If you use the dataset, please cite:
% 
% S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, J. Dubois, 
% Unsupervised skin tissue segmentation for remote photoplethysmography, 
% Pattern Recognition Letters, Elsevier, 2017.
%
% yannick.benezeth@u-bourgogne.fr
clear;
close all;
clc;
% dataset folder
root  =   'C:/Users/parm0/OneDrive/Desktop/preprocess_data/pre_data/';
% get folder list
dirs=dir(root);
j=0;
dirs=dirs(~ismember({dirs.name},{'.','..','desktop.ini'}));
%dirs.name
%Iterate through all directories
all_HR_data = [];
all_PPG_data = [];
all_vid_data = [];
all_facial_vidoes = [];
all_facial_vidoes_norm = [];
for i=1:size(dirs)
    vidFolder   =   [root dirs(i).name];    
    
    % load ground truth
	gtfilename=[vidFolder '/gtdump.xmp']; % DATASET_1
    if exist(gtfilename, 'file')==2
	
		gtdata=csvread(gtfilename);
		gtTrace=gtdata(:,4);
		gtTime=gtdata(:,1)/1000;
		gtHR = gtdata(:,2);
	else 
		gtfilename=[vidFolder '/ground_truth.txt']; %DATASET_2
		
			gtdata=dlmread(gtfilename);
        if exist(gtfilename, 'file')==2   
			gtTrace=gtdata(1,:)';
			gtTime=gtdata(3,:)'; 
			gtHR = gtdata(2,:)';  
        end
    end
	% normalize data (zero mean and unit variance)
	gtTrace = gtTrace - mean(gtTrace,1);
	gtTrace = gtTrace / std(gtTrace);  
	
    all_HR_data = [all_HR_data; gtHR];
    all_PPG_data = [all_PPG_data; gtTrace];
    
    % open video file
    vidObj = VideoReader([ vidFolder '/vid.avi' ]);
    fps = vidObj.FrameRate;
    Duration = vidObj.Duration;
    n=0;
    FramesToRead=ceil(Duration*fps);
    RGB=zeros(FramesToRead,3);
    img_id=1;
    img1=1;
    while hasFrame(vidObj)
        % track frame index
        n=n+1;

        % read frame by frame
        img = readFrame(vidObj);
        file_name1 = num2str(img1);
        img1 = img1+1;
        
        imwrite(img, strcat('C:/Users/parm0/OneDrive/Desktop/preprocess_data/images/img11/', 'Frame' , file_name1, '.jpg'), 'jpg');
        figure(1), imshow(img), title('Frames');
        %to detect face
        FD_obj = vision.CascadeObjectDetector; 
        FD_obj.MergeThreshold=10;
        BB = step(FD_obj, img);
        figure(2), imshow(img), title('Detected Faces');
        for jdx = 1:size(BB,1)
             rectangle('Position',BB(jdx,:),'LineWidth',3,'LineStyle','-','EdgeColor','r');
        end
        for kdx = 1:size(BB,1)
             Fimg = imcrop(img, BB(kdx,:));
             figure(3), subplot(2,2,kdx); imshow(Fimg);
             file_name = num2str(img_id);
             img_id = img_id+1;
             imwrite(Fimg, strcat('C:/Users/parm0/OneDrive/Desktop/preprocess_data/Subject11/', 'Frame' , file_name, '.jpg'), 'jpg');
        end 
    %position for optional skin segmentation
        RGB(n,:) = sum(sum(img));
         
    end
    all_vid_data = [all_vid_data, img];
    all_facial_vidoes = [all_facial_vidoes; RGB];

    RGBNorm=zeros(size(RGB));
    Lambda=100;
    for c=1:3
        RGBDetrend= spdetrend(RGB(:,c),Lambda); %M. P. Tarvainen, TBME, 2002
        RGBNorm(:,c) = (RGBDetrend - mean(RGBDetrend))/std(RGBDetrend); %normalize to zero mean and unit variance
    end
    j = j+1;
    all_facial_vidoes_norm = [all_facial_vidoes_norm; RGBNorm(:,:)];
    fprintf('%d',j);
end
