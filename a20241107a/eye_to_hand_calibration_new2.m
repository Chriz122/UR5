clc,clear,close all

point_num = 132;
list =[1:5];
NUM = length(list);
impoint = ones(NUM*point_num, 4);
XYZ = zeros(NUM, 3);
% cam_location = ones(4*point_num,9);
cam_location = ones(NUM*point_num,3);
a = zeros(132,3);
XYZ_ = XYZ;

% cam_location = load('pose.txt')*1000;
for n=1:11
    a(n*11+1:n*11+11,1)=0.025*n;
end
for n=2:11
    a(n:11:n+11*11,2)=-0.025*(n-1);
end
for n=1:5
    cam_location((n-1)*point_num+1:n*point_num,1:3) = a*1000;
    cam_location((n-1)*point_num+1:n*point_num,3) = cam_location((n-1)*point_num+1:n*point_num,3)+50*(n-1);
end

% for n=1:4
%     cam_location((n-1)*point_num+1:n*point_num,1:9) = load('pose.txt')*1000;
%     cam_location((n-1)*point_num+1:n*point_num,3) = cam_location((n-1)*point_num+1:n*point_num,3)+50*(n-1);
% end


plotCalbPoint3d_noText(cam_location);   
view(-20,90)
figure

%cam_location = cam_location/1000;

count = 0;

for c=list
    count = count + 1;
    pic_name=strcat('A',num2str(c),'.png');
    I=imread(pic_name);
    imagePoints = detectCheckerboardPoints(I);
    %imagePoints = flip(imagePoints);
    %type = check(imagePoints,boardSize);
    %imagePoints = change(imagePoints,boardSize,type);
    %if type ~= 0
        %imagePoints = change(imagePoints,boardSize,type);
    %end

    [x,y]=meshgrid(1:640,1:480);
    txt_name=strcat('D',num2str(c),'.txt');
    depth=load(txt_name);
    imagePoints(:,3)=griddata(x(depth~=0),y(depth~=0),depth(depth~=0),imagePoints(:,1),imagePoints(:,2),'linear');
    
    imagePoints(:,3)=min(imagePoints(:,3));
    
    impoint((count-1)*point_num+1:count*point_num,1:3)=imagePoints;
    
    %%%%%%%%%%%
    

end
impoint(:,1)=impoint(:,1).*impoint(:,3)/10000;
impoint(:,2)=impoint(:,2).*impoint(:,3)/10000;


C=zeros(3,4);
for a=1:3
    C(a,:)=impoint\cam_location(:,a);
end

test = impoint*C.';
tt=cam_location(:,1:3);
t=(tt-test).^2;
T=(sum(t,2)).^0.5;

bar(1:NUM*point_num,T);

title('計算轉換式的點 誤差距離');
xlabel('第幾個點');
ylabel('誤差距離(mm)');

MSE_xyz=sum(t)/length(t)
MSE_distErr=sum(T)/length(T)

%threading