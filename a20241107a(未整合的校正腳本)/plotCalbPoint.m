function  plotCalbPoint(xy)
num=length(xy);
x=xy(:,1);y=-xy(:,2);
plot(x,y,'+');
axis([0 640 -480 0])

for n=1:num
    sn=num2str(n);
    text(x(n),y(n),sn,'FontSize',6)
end