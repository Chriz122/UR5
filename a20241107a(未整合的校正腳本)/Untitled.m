clc,clear,close all
point_num = 132;
list =[1:4];
NUM = length(list);
cam_location = ones(NUM*point_num,3);
a = zeros(132,3);

for n=1:11
    a(n*11+1:n*11+11,1)=0.025*n;
end

for n=2:11
    a(n:11:n+11*11,2)=-0.025*(n-1);
end

for n=1:4
    cam_location((n-1)*point_num+1:n*point_num,1:3) = a*1000;
    cam_location((n-1)*point_num+1:n*point_num,3) = cam_location((n-1)*point_num+1:n*point_num,3)+50*(n-1);
end