clear all;
close all;
clc;
disp('Please browse for Training file');
[file_tr,path_tr] = uigetfile('*.txt');
disp('Now please browse for Test file');
[file_tst,path_tst] = uigetfile('*.txt');
prompt='Please enter number of neighboring point desired (k value)';
k = input(prompt);

%%%%%
train=importdata(file_tr,' '); %Input training file
test=importdata(file_tst,' '); %Input test file
%k=50; %Input "k" value

trainx=train(:,1:17);
testy=test(:,1:17);
x=(train(:,1:16)./max(train(:,1:16)));
y=(test(:,1:16)./max(test(:,1:16)));


mean_arr_X=mean((x));
sd_arr_X=std((x));
X=x-mean_arr_X;
XX=X./sd_arr_X;

mean_arr_Y=mean((y));
sd_arr_Y=std((y));
Y=y-mean_arr_Y;
YY=Y./sd_arr_Y;

train_class=train(:,17);
dist{size(test,1),size(train,1)}=[];

YY_repmat=repmat(permute(YY,[3,2,1]),size(x,1),1);
d=(XX-YY_repmat);
di=d.^2;
dis=sum(di,2);


dist_un=sqrt(sum((XX-YY_repmat).^2,2));
dist=permute(dist_un,[1,3,2]);
[sort_dist,sort_index]=sort(dist,1);%sort distance "dist" in asc order with index
label_dist=sort_dist(1:k,:);%Select only top 'k' training point which are nearest point to test point
label_sort=sort_index(1:k,:);

label_class=reshape(trainx(label_sort,17),k,size(testy,1));%Reshaping the label_sort to 2D array with 'k' rows
if(k~=1)
[pred_class_un]=histc(label_class,unique(label_class));%Histogram of the label_class
else
    [pred_class_un]=label_class;
end
pred_class_count=[];
pred_class=[];
for i =1:size(y,1)
    [pred_class_count(i),pred_class(i)]=max(pred_class_un(:,i));
    if (find(pred_class_un(:,i)==pred_class_count(i)))
        [pred_class_row,pred_class_col]=find(pred_class_un(:,i)==pred_class_count(i));
        pred_class(i)=pred_class_row(randi(size(pred_class_row,1),1,1));
    end
end



classification_accuracy=(sum(pred_class-(test(:,17)+1)'==0)./size(test,1))*100;

for i =1:size(pred_class,2)
    
fprintf('Object ID %d, Predicted Class %d, True Class %d, Classification Accuracy= %.6f, \n',i, pred_class(i)-1,test(i,17),classification_accuracy);
end

