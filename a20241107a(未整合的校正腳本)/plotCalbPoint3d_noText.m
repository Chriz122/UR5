function  plotCalbPoint3d(xy)
num=length(xy);
y=xy(:,1);x=-xy(:,2);z=xy(:,3);
plot3(x,y,z,'+');
xmin=min(x);xmax=max(x);
ymin=min(y);ymax=max(y);
zmin=min(z);zmax=max(z);
axis([xmin-5 xmax+5 ymin-5 ymax+5 zmin-5 zmax+5])

grid on
for n=1:num
    sn=num2str(n);
    text(x(n),y(n),z(n),sn)
end

xlabel('Y'),ylabel('X'),zlabel('Z')