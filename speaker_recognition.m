>> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%�����������ڵ��˵�˵����ȷ��
clear all;
close all;
MFCC_size=12;%mfcc��ά��
GMMM_component=16;%GMM component ����

mu_model=zeros(MFCC_size,GMMM_component);%��˹ģ�� ���� ��ֵ
sigma_model=zeros(MFCC_size,GMMM_component);%��˹ģ�� ���� ����
weight_model=zeros(GMMM_component);%��˹ģ�� ���� Ȩ��

train_file_path='.\training\';%ģ��ѵ���ļ�·��
test_file_path='.\testing\';%�����ļ�·��

all_train_feature=[];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%train model
FileList=dir(train_file_path);%��ȡ��·���µ������ļ�
model_num=1;%ע��ģ�͵ĸ���
%��·�����Ƿ����ļ���
for i=1:length(FileList)
    if(FileList(i).isdir==1&&~strcmp(FileList(i).name,'.')&&~strcmp(FileList(i).name,'..'))
        all_model_name{model_num, 1}=FileList(i).name;%�洢ģ������ 
        fprintf('Train:%s\n',all_model_name{model_num, 1});
        one_train_file_path=[train_file_path  all_model_name{model_num, 1} '\'];
        all_train_file=dir(fullfile(one_train_file_path,'/*.wav'));%��ȡ��·���µ������ļ�
        all_train_feature = [];
        for j=1:length(all_train_file)
            file_name=all_train_file(j).name;%wav�ļ���
            train_file=[one_train_file_path file_name];
            %fprintf('  train file:%s\n',train_file);
            [wav_data ,fs]=audioread(train_file);
            train_feature=melcepst(wav_data ,fs);
            all_train_feature=[all_train_feature;train_feature];
        end
        dirName=['.\model\' all_model_name{model_num, 1} '\'];
        [mu_model,sigma_model,weight_model]=gmm_estimate(all_train_feature',GMMM_component);
        if ~exist( dirName, 'dir')
            mkdir(dirName);
        end
        save([dirName 'mu_model.mat'],'mu_model');
        save([dirName 'sigma_model.mat'],'sigma_model');
        save([dirName 'weight_model.mat'],'weight_model');
        model_num=model_num+1;
    end
end
save('.\model\all_model_name.mat','all_model_name');

all_model_name=importdata('.\model\all_model_name.mat');
model_num=length(all_model_name);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%test
FileList=dir(test_file_path);%��ȡ��·���µ������ļ�
%��·�����Ƿ����ļ���
count = 0;
for i=1:length(FileList)
    if(FileList(i).isdir==1&&~strcmp(FileList(i).name,'.')&&~strcmp(FileList(i).name,'..'))
        test_name=FileList(i).name;
        one_test_file_path=[test_file_path  test_name '\'];
        all_test_file=dir(fullfile(one_test_file_path,'/*.wav'));%��ȡ��·���µ������ļ�
        fprintf('�������ͣ�%s\n',test_name);
        for j=1:length(all_test_file)
            file_name=all_test_file(j).name;%wav�ļ���
            test_file=[one_test_file_path file_name];
            [wav_data ,fs]=audioread(test_file);
            test_feature=melcepst(wav_data ,fs);
            fprintf('Test��%s\n',test_file);
            for k=1:model_num
                model_path=['.\model\' all_model_name{k, 1} '\'];
                mu_model=importdata([model_path 'mu_model.mat']);
                sigma_model=importdata([model_path 'sigma_model.mat']);
                weight_model=importdata([model_path 'weight_model.mat']);
                [lYM, lY] = lmultigauss(test_feature', mu_model, sigma_model, weight_model);
                score(j,k) = mean(lY);
                %fprintf('   Model:%s  score:%f\n',all_model_name{k, 1},score(j,k));
            end
            [max_score,max_id]=max(score(j,:));
            [min_score,min_id]=min(score(j,:));
            if(max_id == 24)
                count = count + 1;
            end
            fprintf('Max score:%f  model:%s\nMin score:%f  model:%s\n\n',max_score,all_model_name{max_id, 1},min_score,all_model_name{min_id, 1});
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %result
        fprintf('Correct Number:%d.\n',count);
    end
end